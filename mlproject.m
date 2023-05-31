filename = 'wine.csv';
data = readtable(filename);
class = data.Var1;
a_data = table2array(data);
winefeatures = dataset(data.Var2,data.Var3,data.Var4,data.Var5,data.Var6,data.Var7,data.Var8,data.Var9,data.Var10,data.Var11,data.Var12,data.Var13,data.Var14);
t_winefeatures = dataset2table(winefeatures);
a_winefeatures = table2array(t_winefeatures);
min_deger = min(a_winefeatures);
max_deger = max(a_winefeatures);
n_winefeatures = (a_winefeatures - min_deger) ./ (max_deger - min_deger);

mrmr_winefeatures = fscmrmr(n_winefeatures,class);

correlation = abs(corr(n_winefeatures,class));
threshold = 0.5;
selected_features = find(correlation > threshold);
nfs_winefeatures = n_winefeatures(:,[4 6 7 11 12 13]);

holdout = cvpartition(class,'HoldOut',0.3);
data_train = nfs_winefeatures(holdout.training,:);
data_test = nfs_winefeatures(holdout.test,:);
class_train = class(holdout.training);
class_test = class(holdout.test);
 
k = 5;

knn_model = fitcknn(data_train,class_train,'NumNeighbors',k);
result = predict(knn_model,data_test);

cm_knn = confusionchart(class_test,result);

tree_model = fitctree(data_train,class_train);
tree_predict = predict(tree_model,data_test);

cm_tree = confusionchart(class_test,tree_predict);

