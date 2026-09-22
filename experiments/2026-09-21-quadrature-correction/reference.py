"""Float64 path-dependent SHAP reference with general-order Gauss-Legendre rules.

Loads exact float32 model fields from JSON, retaining multiclass tree groups and
categorical/missing routing. Entire traversal is JIT compiled; no C++ SHAP calls.
"""
import json
import numpy as np
from numba import njit, prange

@njit
def visit(node, row, nodes, weights, left, right, feature, threshold, default_left,
          cat_start, cats, value, cover, basis, weight_prod, path, out):
    f = feature[node]
    if left[node] < 0:
        return basis * (value[node] * weight_prod)
    l, r = left[node], right[node]
    v = row[f]
    if np.isnan(v) or (cat_start[node+1] > cat_start[node] and v < 0):
        hot = l if default_left[node] else r
    elif cat_start[node+1] > cat_start[node]:
        match = False
        for k in range(cat_start[node], cat_start[node+1]):
            if int(v) == cats[k]:
                match = True
        hot = r if match else l
    else:
        hot = l if v < threshold[node] else r
    total = np.zeros(len(nodes))
    old = path[f]
    for child in (l, r):
        cw = cover[child] / cover[node]
        if cw == 0:
            continue
        up = 1.0 if np.isnan(old) else old
        edge = up / cw if child == hot else 0.0
        child_basis = basis * (1.0 + (edge - 1.0) * nodes)
        if not np.isnan(old) and old != 1.0:
            child_basis /= 1.0 + (old - 1.0) * nodes
        path[f] = edge
        h = visit(child, row, nodes, weights, left, right, feature, threshold,
                  default_left, cat_start, cats, value, cover, child_basis,
                  weight_prod*cw, path, out)
        for q in range(len(nodes)):
            delta = (edge-1)/(1+(edge-1)*nodes[q]) - (up-1)/(1+(up-1)*nodes[q])
            out[f] += weights[q] * h[q] * delta
        total += h
    path[f] = old
    return total

@njit(parallel=True)
def explain_rows(x, roots, groups, ngroups, nodes, weights, left, right, feature,
                 threshold, default_left, cat_start, cats, value, cover):
    out = np.zeros((len(x), ngroups, x.shape[1]))
    for i in prange(len(x)):
        path = np.full(x.shape[1], np.nan)
        for t in range(len(roots)):
            visit(roots[t], x[i], nodes, weights, left, right, feature, threshold,
                  default_left, cat_start, cats, value, cover, np.ones(len(nodes)),
                  1.0, path, out[i, groups[t]])
    return out

class Reference:
    def __init__(self, booster):
        model = json.loads(booster.save_raw(raw_format='json'))['learner']
        forest = model['gradient_booster']['model']
        groups = np.asarray(forest['tree_info'], dtype=np.int64)
        self.ngroups = int(groups.max()) + 1
        self.roots = []
        fields = {k: [] for k in ['left','right','feature','threshold','default_left','value','cover']}
        cats, starts = [], [0]
        self.max_depth = 0
        self.max_unique_depth = 0
        for tree in forest['trees']:
            offset = len(fields['left'])
            self.roots.append(offset)
            n = len(tree['left_children'])
            for k, source in [('left','left_children'),('right','right_children')]:
                fields[k].extend([c+offset if c >= 0 else -1 for c in tree[source]])
            fields['feature'].extend(tree['split_indices'])
            fields['threshold'].extend(tree['split_conditions'])
            fields['value'].extend(tree['split_conditions'])
            fields['cover'].extend(tree['sum_hessian'])
            fields['default_left'].extend(tree['default_left'])
            cat_map = {}
            for node, start, size in zip(tree['categories_nodes'],tree['categories_segments'],tree['categories_sizes']):
                cat_map[node] = tree['categories'][start:start+size]
            for node in range(n):
                cats.extend(cat_map.get(node, []))
                starts.append(len(cats))
            def depth(node, seen, d):
                if tree['left_children'][node] < 0:
                    self.max_depth = max(self.max_depth, d)
                    self.max_unique_depth = max(self.max_unique_depth, len(seen))
                    return
                seen = seen | {tree['split_indices'][node]}
                depth(tree['left_children'][node],seen,d+1)
                depth(tree['right_children'][node],seen,d+1)
            depth(0,set(),0)
        self.arr = []
        for k in fields:
            if k in ['threshold','value','cover']:
                a = np.array(fields[k], dtype=np.float32).astype(np.float64)
            else:
                a = np.array(fields[k], dtype=np.int64)
            self.arr.append(a)
        self.cats = np.asarray(cats, dtype=np.int64)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.groups = groups
        self.roots = np.asarray(self.roots,dtype=np.int64)

    def explain(self, x, points):
        nodes, weights = np.polynomial.legendre.leggauss(points)
        left,right,feature,threshold,default_left,value,cover = self.arr
        return explain_rows(np.ascontiguousarray(x,dtype=np.float64),self.roots,self.groups,
                            self.ngroups,(nodes+1)/2,weights/2,left,right,feature,threshold,
                            default_left,self.starts,self.cats,value,cover)

def numeric_rows(x):
    if hasattr(x, 'dtypes'):
        x = x.copy()
        for c in x.columns:
            if str(x[c].dtype) == 'category':
                x[c] = x[c].cat.codes.astype(float).replace(-1,np.nan)
    return np.asarray(x,dtype=np.float64)

@njit
def visit_interactions(node, row, nodes, weights, left, right, feature, threshold,
                       default_left, cat_start, cats, value, cover, z, o, seen,
                       active, depth, phi, out):
    if left[node] < 0:
        ratios=np.empty(depth)
        for q in range(len(nodes)):
            scale=value[node]*weights[q]
            for i in range(depth):
                f=active[i]
                factor=z[f]+(o[f]-z[f])*nodes[q]
                scale*=factor
                ratios[i]=(o[f]-z[f])/factor
            for i in range(depth):
                fi=active[i]
                phi[fi]+=scale*ratios[i]
                for j in range(i):
                    fj=active[j]
                    term=0.5*scale*ratios[i]*ratios[j]
                    out[fi,fj]+=term
                    out[fj,fi]+=term
        return
    f=feature[node]
    l,r=left[node],right[node]
    v=row[f]
    if np.isnan(v) or (cat_start[node+1]>cat_start[node] and v<0):
        hot=l if default_left[node] else r
    elif cat_start[node+1]>cat_start[node]:
        match=False
        for k in range(cat_start[node],cat_start[node+1]):
            if int(v)==cats[k]:
                match=True
        hot=r if match else l
    else:
        hot=l if v<threshold[node] else r
    old_z,old_o=z[f],o[f]
    was_seen=seen[f]
    if not was_seen:
        active[depth]=f
        depth+=1
        seen[f]=True
    for child in (l,r):
        cw=cover[child]/cover[node]
        if cw==0:
            continue
        z[f]=old_z*cw
        o[f]=old_o if child==hot else 0.0
        visit_interactions(child,row,nodes,weights,left,right,feature,threshold,
                           default_left,cat_start,cats,value,cover,z,o,seen,active,
                           depth,phi,out)
    z[f],o[f]=old_z,old_o
    seen[f]=was_seen

@njit
def interaction_rows(x, roots, groups, ngroups, nodes, weights, left, right,
                     feature, threshold, default_left, cat_start, cats, value, cover):
    out=np.zeros((len(x),ngroups,x.shape[1],x.shape[1]))
    phi=np.zeros((len(x),ngroups,x.shape[1]))
    for i in range(len(x)):
        z=np.ones(x.shape[1]); o=np.ones(x.shape[1])
        seen=np.zeros(x.shape[1],dtype=np.bool_)
        active=np.empty(x.shape[1],dtype=np.int64)
        for t in range(len(roots)):
            visit_interactions(roots[t],x[i],nodes,weights,left,right,feature,threshold,
                               default_left,cat_start,cats,value,cover,z,o,seen,active,0,
                               phi[i,groups[t]],out[i,groups[t]])
        for g in range(ngroups):
            for f in range(x.shape[1]):
                out[i,g,f,f]=phi[i,g,f]-np.sum(out[i,g,f])
    return out

def interactions(ref, x, points):
    nodes,weights=np.polynomial.legendre.leggauss(points)
    left,right,feature,threshold,default_left,value,cover=ref.arr
    return interaction_rows(np.ascontiguousarray(x,dtype=np.float64),ref.roots,ref.groups,
                            ref.ngroups,(nodes+1)/2,weights/2,left,right,feature,threshold,
                            default_left,ref.starts,ref.cats,value,cover)
