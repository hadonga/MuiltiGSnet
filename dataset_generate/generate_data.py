import numpy as np
import open3d as o3d
import os
from multiprocessing import Process
from tqdm import tqdm
import yaml

'''
2021.1.19 生成数据及标签

生成按物理性质采样后的点云数据以及标签
clip_sn_pc:按物理性质采样后得到的点云数据
pl_cl_s_lb：用于训练的标签
plpt_cl_s_lb：用于测试准确度的标签
sn_lb:每个pillar的法向量标签

生成没有经过物理性质采样后的点云数据以及标签
clip_n_pc:未进行物理性质采样的点云数据
pl_cl_lb：用于训练的标签
plpt_cl_lb：用于测试准确度的标签
n_lb:每个pillar的法向量标签
'''

config_file = './config_kittiSem.yaml'

try:
    with open(config_file) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    print("using config file:", config_file)
    print('\n'.join('%s:%s' % item for item in config_dict.items()))


    class ConfigClass:
        def __init__(self, **entries):
            self.__dict__.update(entries)


    cfg = ConfigClass(**config_dict)
except:
    print("Error!!! => no config file found at '{}'".format(config_file))

root_path = cfg.root_dir

class bin():
    def __init__(self):
        self.has_point = False
        self.num_g = 0
        self.num_ng = 0
        self.clf = 0
        self.has_normal = False
        self.pcd = o3d.geometry.PointCloud()
        self.normal = [0, 0, 1]
        self.points = []

    def add_point(self,p):
        if not self.has_point:
            self.has_point = True
        self.num_g += 1
        self.points.append(p)
        '''
        if self.has_point:
            if self.clf==3:
                self.clf=2
        else:
            self.clf = 1
            self.has_point = True
        '''

    def add_non_point(self):
        if not self.has_point:
            self.has_point = True
        self.num_ng += 1
        '''
        if self.has_point:
            if self.clf==1:
                self.clf=2
        else:
            self.clf=3
            self.has_point=True
        '''

    def get_has_normal(self):
        if len(self.points)>5:
            self.has_normal=True

    def get_clf(self):
        if self.has_point:
            if self.num_ng >= self.num_g:
                self.clf = 2
            else:
                self.clf = 1

    def get_normal(self):
        '''
        if len(self.points)>5:
            #self.has_normal = True
            cloud=np.array(self.points)
            X = np.ones([len(cloud), 3])
            X[:, :2] = cloud[:, :2]
            Z = cloud[:, 2:3]
            X1 = np.dot(X.T, X)
            X2=np.linalg.inv(X1)
            X3 = np.dot(X2, X.T)
            A = np.dot(X3, Z)
            a,b,d=A
            c=-1
            self.normal = np.array([(a / ((a * a + b * b + c * c) ** 0.5)), (b / ((a * a + b * b + c * c) ** 0.5)),
                                    (c / ((a * a + b * b + c * c) ** 0.5))])
        '''

        if (len(self.points) > 5):
            self.has_normal = True
            self.pcd.points = o3d.utility.Vector3dVector(self.points)
            plane_model, inliers = self.pcd.segment_plane(distance_threshold=0.02,
                                                          ransac_n=len(self.points),
                                                          num_iterations=2)
            [a, b, c, d] = plane_model
            self.normal = np.array([(a / ((a * a + b * b + c * c) ** 0.5)), (b / ((a * a + b * b + c * c) ** 0.5)),
                                    (c / ((a * a + b * b + c * c) ** 0.5))])

'''网格图的上限'''
def limit_up(i, counter):
    if counter + i + 1 > 127:
        return 127
    else:
        return counter + i + 1

'''网格图的下限'''
def limit_dn(i, counter):
    if i - counter < 0:
        return 0
    else:
        return i - counter

'''得到pillar中有点，但地面点数量少于5个的pillar的法向量'''
def fill_normal(grid,i,j):
    counter=1
    has_normal=False
    pcd=[]
    cloud=o3d.geometry.PointCloud()
    while not has_normal:
        pcd.clear()
        for i_i in range(limit_dn(i, counter), limit_up(i, counter)):
            for i_j in range(limit_dn(j, counter), limit_up(j, counter)):
                if grid[i_i][i_j].has_point:
                    pcd += grid[i_i][i_j].points
        counter = counter + 1
        '''
        if len(pcd)>5:
            has_normal = True
            cloud=np.array(pcd)
            X = np.ones([len(cloud), 3])
            X[:, :2] = cloud[:, :2]
            Z = cloud[:, 2:3]
            X1 = np.dot(X.T, X)
            X2=np.linalg.inv(X1)
            X3 = np.dot(X2, X.T)
            A = np.dot(X3, Z)
            a,b,d=A
            c=-1
            return np.array([(a / ((a * a + b * b + c * c) ** 0.5)), (b / ((a * a + b * b + c * c) ** 0.5)),
                                    (c / ((a * a + b * b + c * c) ** 0.5))])
        '''
        if (len(pcd) > 5):
            has_normal = True
            cloud.points = o3d.utility.Vector3dVector(np.array(pcd))
            plane_model, inliers = cloud.segment_plane(distance_threshold=0.02,
                                                          ransac_n=5,
                                                          num_iterations=5)
            [a, b, c, d] = plane_model
            return np.array([(a / ((a * a + b * b + c * c) ** 0.5)), (b / ((a * a + b * b + c * c) ** 0.5)),
                                    (c / ((a * a + b * b + c * c) ** 0.5))])


def get_index(point):
    temp=max(abs(point[0]),abs(point[1]))
    return int(np.floor(temp/0.5))

def my_function(points):
    counter = np.zeros([104, 1])
    max_num=0
    for i in range(len(points)):
        index = get_index(points[i])
        counter[int(index)] = counter[int(index)] + 1
        if max_num<counter[int(index)]:
            max_num=counter[int(index)]
    counter=max_num*2-counter
    return counter

def my_probability(points):
    p=np.zeros([len(points),],dtype=float)#初始化为0，后续存储每个点的概率
    counter=my_function(points) #划分各个区域，返回每个区域点的个数，区域为方形环，固定宽度0.5m
    for i in range(len(points)):
        p[i]=counter[get_index(points[i])]
    p=p/p.sum()
    return p

class ground_grid():
    def __init__(self, data_path, label_path):
        self.cloud = []
        self.data_path = data_path
        self.label_path = label_path
        self.root_path = root_path
        self.acc_lb=[]
        self.map_label = np.zeros([128, 128])
        self.pcd = o3d.geometry.PointCloud()
        self.ground=[]
        self.normal = np.zeros([128, 128, 3])  # [y,x,n]
        for i in range(128):
            self.ground.append([])
            for j in range(128):
                ground_bin = bin()
                self.ground[i].append(ground_bin)

    def in_put(self):
        data = np.fromfile(self.data_path, dtype=np.float32).reshape(-1, 4)
        label = np.fromfile(self.label_path, dtype=np.uint32).reshape((-1))
        data = np.vstack((data.T, label)).T
        #num_g = 0
        #num_non_g = 0
        #data = np.array([x for x in data if (0 < x[0] + 51.2 < 102.4 and 0 < x[1] + 51.2 < 102.4)])
        for x in data:
            if (0 < x[0] + 51.2 < 102.4 and 0 < x[1] + 51.2 < 102.4):
                x[2]+=1.733
                self.cloud.append(x[:5])
                self.acc_lb.append(x[4])
                '''
                if x[4] == 40 or x[4] == 44 or x[4] == 48 or x[4] == 49 or x[4] == 60:  # 地面点判断
                    grid_index = (x[:2] + 51.2) / 0.8
                    grid_index = np.floor(grid_index)
                    self.ground[int(grid_index[1])][int(grid_index[0])].add_point()
                    #num_g = num_g + 1
                else:
                    grid_index = (x[:2] + 51.2) / 0.8
                    grid_index = np.floor(grid_index)
                    self.ground[int(grid_index[1])][int(grid_index[0])].add_non_point()  # 加非地面点
                    #num_non_g += 1
                '''
        # 调用o3d函数，得到法向量，再在原数据上拼接法向量，得到nX7的n_pc
        self.cloud = np.array(self.cloud)
        self.points = self.cloud[:, :3]
        self.pcd.points = o3d.utility.Vector3dVector(self.points)
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
        self.all_normal = np.array(self.pcd.normals)
        self.acc_lb = np.array(self.acc_lb)
        self.n_pc = np.vstack((self.cloud[:, :4].T, self.all_normal.T)).T

        if cfg.physical_sampling:
            if len(self.cloud)>=100000:
                choice = np.random.choice(len(self.cloud), 100000, replace=False, p=my_probability(self.cloud))
                self.s_pc=self.cloud[choice]
                self.l_s=self.acc_lb[choice]
                self.s_n=self.all_normal[choice]
                self.sn_pc = np.vstack((self.s_pc[:,:4].T, self.s_n.T)).T
                for x in self.s_pc:
                    if x[4] == 40 or x[4] == 44 or x[4] == 48 or x[4] == 49 or x[4] == 60:  # 地面点判断
                        grid_index = (x[:2] + 51.2) / 0.8
                        grid_index = np.floor(grid_index)
                        self.ground[int(grid_index[1])][int(grid_index[0])].add_point(x[:3])
                    # num_g = num_g + 1
                    else:
                        grid_index = (x[:2] + 51.2) / 0.8
                        grid_index = np.floor(grid_index)
                        self.ground[int(grid_index[1])][int(grid_index[0])].add_non_point()  # 加非地面点
                self.deal_ground()
                self.save_result()
        else:
            for x in self.cloud:
                if x[4] == 40 or x[4] == 44 or x[4] == 48 or x[4] == 49 or x[4] == 60:  # 地面点判断
                    grid_index = (x[:2] + 51.2) / 0.8
                    grid_index = np.floor(grid_index)
                    self.ground[int(grid_index[1])][int(grid_index[0])].add_point(x[:3])
                # num_g = num_g + 1
                else:
                    grid_index = (x[:2] + 51.2) / 0.8
                    grid_index = np.floor(grid_index)
                    self.ground[int(grid_index[1])][int(grid_index[0])].add_non_point()  # 加非地面点
            self.deal_ground()
            self.save_result()

        #print("地面点：", num_g, "----非地面点：", num_non_g)

    def deal_ground(self):
        for i in range(128):
            for j in range(128):
                self.ground[i][j].get_clf()
                self.map_label[i, j] = self.ground[i][j].clf
                # 处理得到法向量
                self.ground[i][j].get_has_normal()
                if self.ground[i][j].has_normal:
                    self.ground[i][j].get_normal()
                    self.normal[i, j] = self.ground[i][j].normal
                elif not self.ground[i][j].has_normal and self.ground[i][j].has_point:
                    self.normal[i, j] = fill_normal(self.ground, i, j)


    def save_result(self):
        part = self.data_path.split('/')[-3]
        index = self.data_path.split('/')[-1]
        index = index[:-4]
        #存储数据
        if cfg.physical_sampling:
            np.save(os.path.join(self.root_path, part, 'clip_sn_pc', index), np.array(self.sn_pc))
            np.save(os.path.join(self.root_path, part, 'pl_cl_s_lb', index), np.array(self.map_label))
            np.save(os.path.join(self.root_path, part, 'plpt_cl_s_lb', index), np.array(self.l_s))
            np.save(os.path.join(self.root_path, part, 'sn_lb', index), np.array(self.normal))
        else:
            np.save(os.path.join(self.root_path, part, 'plpt_cl_lb', index), np.array(self.acc_lb))
            np.save(os.path.join(self.root_path, part, 'clip_n_pc', index), np.array(self.n_pc))
            np.save(os.path.join(self.root_path, part, 'pl_cl_lb', index), np.array(self.map_label))
            np.save(os.path.join(self.root_path, part, 'n_lb', index), np.array(self.normal))

class my_process(Process):
    def __init__(self, num):
        super(my_process, self).__init__()
        self.num = num

    def run(self):
        counter = 0
        #discard=0
        while counter*worker+self.num<len(d_dir):
            ground = ground_grid(d_dir[counter*worker+self.num], l_dir[counter*worker+self.num])
            ground.in_put()
            if counter % 10 == 0:
                print("worker_%d:complete data:"%self.num, counter)
                #print("worker_%d:discard data:"%self.num, discard)
            counter=counter+1
#创建目录
def pre_deal(root_path):
    if cfg.physical_sampling:
        if not os.path.exists(os.path.join(root_path, 'clip_sn_pc')):
            os.mkdir(os.path.join(root_path, 'clip_sn_pc'))
        if not os.path.exists(os.path.join(root_path, 'pl_cl_s_lb')):
            os.mkdir(os.path.join(root_path, 'pl_cl_s_lb'))
        if not os.path.exists(os.path.join(root_path, 'plpt_cl_s_lb')):
            os.mkdir(os.path.join(root_path, 'plpt_cl_s_lb'))
        if not os.path.exists(os.path.join(root_path, 'sn_lb')):
            os.mkdir(os.path.join(root_path, 'sn_lb'))
    else:
        if not os.path.exists(os.path.join(root_path, 'clip_n_pc')):
            os.mkdir(os.path.join(root_path, 'clip_n_pc'))
        if not os.path.exists(os.path.join(root_path, 'pl_cl_lb')):
            os.mkdir(os.path.join(root_path, 'pl_cl_lb'))
        if not os.path.exists(os.path.join(root_path, 'plpt_cl_lb')):
            os.mkdir(os.path.join(root_path, 'plpt_cl_lb'))
        if not os.path.exists(os.path.join(root_path, 'n_lb')):
            os.mkdir(os.path.join(root_path, 'n_lb'))


print("gen_clip_sn data product starts")

part=['00','01','02','03','04','05','06','07','08','09','10']
part_length = {'00': 4541, '01': 1101, '02': 4661, '03': 801, '04': 271, '05': 2761,
                   '06': 1101, '07': 1101, '08': 4071, '09': 1591, '10': 1201}
d_dir=[]
l_dir=[]

for part_i in part:
    pre_deal(os.path.join(root_path,part_i))
    for index in range(part_length[part_i]):
        d_dir.append(os.path.join(root_path,part_i,'velodyne','%06d.bin'%index))
        l_dir.append(os.path.join(root_path,part_i,'labels','%06d.label'%index))

process_list = []
worker=8
for num in range(worker):
    pr = my_process(num)
    pr.start()
    process_list.append(pr)

for pr in process_list:
    pr.join()

print("gen_clip_sn data product complete!")
