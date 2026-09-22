"""Check the float64 reference against exhaustive coalition SHAP on small trees."""
import math
import numpy as np
import xgboost as xgb
from reference import Reference, numeric_rows, interactions


def exhaustive(ref, row):
    left,right,feature,threshold,defaults,value,cover=ref.arr
    f=len(row)
    def v(node,mask):
        if left[node]<0:
            return value[node]
        col=feature[node]
        if mask & (1<<col):
            val=row[col]
            categories=ref.cats[ref.starts[node]:ref.starts[node+1]]
            if np.isnan(val) or (len(categories)>0 and val<0):
                child=left[node] if defaults[node] else right[node]
            elif len(categories)>0:
                child=right[node] if val in categories else left[node]
            else:
                child=left[node] if val<threshold[node] else right[node]
            return v(child,mask)
        return (cover[left[node]]*v(left[node],mask)+cover[right[node]]*v(right[node],mask))/cover[node]
    coalitions=np.zeros((1<<f,ref.ngroups))
    for mask in range(1<<f):
        for root,group in zip(ref.roots,ref.groups):
            coalitions[mask,group]+=v(root,mask)
    phi=np.zeros((ref.ngroups,f))
    for i in range(f):
        for mask in range(1<<f):
            if not mask & (1<<i):
                s=mask.bit_count()
                weight=math.factorial(s)*math.factorial(f-s-1)/math.factorial(f)
                phi[:,i]+=weight*(coalitions[mask|(1<<i)]-coalitions[mask])
    interaction=np.zeros((ref.ngroups,f,f))
    for i in range(f):
        for j in range(i):
            for mask in range(1<<f):
                if not mask & ((1<<i)|(1<<j)):
                    s=mask.bit_count()
                    weight=math.factorial(s)*math.factorial(f-s-2)/(2*math.factorial(f-1))
                    interaction[:,i,j]+=weight*(coalitions[mask|(1<<i)|(1<<j)]-coalitions[mask|(1<<i)]-coalitions[mask|(1<<j)]+coalitions[mask])
            interaction[:,j,i]=interaction[:,i,j]
    for i in range(f):
        interaction[:,i,i]=phi[:,i]-interaction[:,i,:].sum(axis=-1)
    return phi,interaction

if __name__=='__main__':
    import pandas as pd
    rng=np.random.RandomState(18)
    for categorical in [False,True]:
        x=pd.DataFrame(rng.normal(size=(256,4)))
        if categorical:
            x[0]=pd.Categorical(rng.randint(0,4,256))
        x.iloc[::11,1]=np.nan
        y=rng.randint(0,3,256)
        dm=xgb.DMatrix(x,label=y,enable_categorical=True)
        model=xgb.train({'objective':'multi:softprob','num_class':3,'max_depth':5,'nthread':2,'tree_method':'hist','max_cat_to_onehot':1},dm,3)
        ref=Reference(model)
        rows=numeric_rows(x.iloc[:3])
        pred=ref.explain(rows,8)
        pair=interactions(ref,rows,8)
        for i,row in enumerate(rows):
            exact_phi,exact_pair=exhaustive(ref,row)
            np.testing.assert_allclose(pred[i],exact_phi,rtol=0,atol=2e-13)
            np.testing.assert_allclose(pair[i],exact_pair,rtol=0,atol=2e-13)
        model.set_param({'shap_algorithm':'quadratureshap','quadratureshap_points':8})
        native=model.predict(xgb.DMatrix(x.iloc[:3],enable_categorical=True),pred_contribs=True)
        np.testing.assert_allclose(pred,native[:,:,:-1],rtol=0,atol=2e-6)
        native_pair=model.predict(xgb.DMatrix(x.iloc[:3],enable_categorical=True),pred_interactions=True)
        np.testing.assert_allclose(pair,native_pair[:,:,:-1,:-1],rtol=0,atol=2e-6)
        print('PASS first/second-order exhaustive coalitions, repeats, missing, multiclass, categorical=',categorical,flush=True)
