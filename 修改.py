print("feature_map: ",feature_map.size()," support: ",support.size()," query:",query.size()," resolution:",resolution)

#--------------调试记录-----------------#

输入：
print("feature_map: ",feature_map.size()," support: ",support.size()," query:",query.size()," resolution:",resolution)
输出：
feature_map:  torch.Size([100, 25, 640])  support:  torch.Size([5, 125, 640])  query: torch.Size([1875, 640])  resolution: 25

#----------------------------


输入：
print("way: ",way," shot: ",shot," d:",d)
输出：
way:  5  shot:  5  d: 640


#-----------------怎么改
args.opt='sgd'
args.lr=1e-1
args.gamma=1e-1
args.epoch=150
args.decay_epoch=70,120
args.val_epoch=5
args.weight_decay=5e-4
args.nesterov=True
args.train_transform_type=0
args.resnet=True
args.train_shot=5 # 
args.train_way=5 # 原本是20
args.test_shot=1,5
args.pre=True
args.gpu=0

# FRN函数内部
return_support=true
inp.size() [100, 3, 84, 84]
feature_map [100, 25, 640]
support [5, 125, 640]
query [1875, 640] # [5,]
train_query_shot 15
shots [5,15]  # shots = [args.train_shot, args.train_query_shot]

recon_dist [3750, 10]
neg_l2_dist [150, 10]


shot=self.shots[0],
query_shot=self.shots[1],

feature_map = self.get_feature_map(inp)
support = feature_map[:way*shot].view(way, shot*resolution , d)
query = feature_map[way*shot:].view(way*query_shot*resolution, d)


recon_dist = self.get_recon_dist(query=query,support=support,alpha=alpha,beta=beta) # way*query_shot*resolution, way
neg_l2_dist = recon_dist.neg().view(way*query_shot,resolution,way).mean(1) # way*query_shot, way

frn的spport是
feature_map [100, 640, 25]
support = feature_map[:way*shot].view(way, shot*resolution , d)
support [5, 125, 640],
glofa的support是
feature_map [100, 640, 5, 5]
support=feature_map[:(train_way*train_shot)].view((train_way*train_shot),640,-1)

在修改frn时，不修改frn函数的接口
在glofa中，是计算出属于每个类的概率
而在frn中，是计算出到每个类的距离

FRN.py的get_recon_dist_glofa()改完了，现在的问题是在model中添加了两个参数，
怎么加载预训练模型
