import numpy as np
import cv2

class Canvas:
    def __init__(self, num_tri=128, size=(256, 256)):
        coords = np.mgrid[0:size[0], 0:size[1]]
        self.coords = coords
        self.size = size
        self.num_tri = num_tri
        # (N, 3, 3)
        position = np.random.rand(num_tri, 3, 3)
        position[:, :, 2] = 1
        self.position = position
        #(only consider RGB, not for Alpha)
        colors = np.random.rand(num_tri, 3)
        self.colors = colors
        self.match_rate = np.inf

    def draw(self):
        count_buffer = np.ones(self.size)
        rgb_buffer = np.zeros((self.size[0], self.size[1], 3))
        for nf in range(self.position.shape[0]):
            position = self.position[nf].copy()
            position[:, 0] *= self.size[0] - 1
            position[:, 1] *= self.size[1] - 1
            v0, v1, v2 = position
            # 左闭右开
            xmin = int(max(0,            min(v0[0], v1[0], v2[0])))
            xmax = int(min(self.size[0], max(v0[0], v1[0], v2[0])+1))
            ymin = int(max(0,            min(v0[1], v1[1], v2[1])))
            ymax = int(min(self.size[1], max(v0[1], v1[1], v2[1])+1))
            # P:(2, nPointsInTriangleInImage)
            P = self.coords[:, xmin:xmax, ymin:ymax].reshape(2,-1)
            # B: is the barycentric coordinates of points inside the triangle bounding box
            B = np.linalg.inv(position.T) @ \
                np.vstack((P, np.ones((1, P.shape[1]))))
            # Cartesian coordinates of points inside the triangle
            I = np.where(np.all(B>=0, axis=0))[0]
            X, Y = P[0,I], P[1,I]
            X, Y = X.astype(np.int32), Y.astype(np.int32)
            count_buffer[Y, X] += 1
            rgb_buffer[Y, X] += self.colors[nf]
        rgb_buffer = rgb_buffer/count_buffer[:,:,None]
        return rgb_buffer, count_buffer

    def calc_match_rate(self, ref_img):
        rgb, count = self.draw()
        rgb_loss = np.abs(ref_img - rgb).sum()
        self.match_rate = rgb_loss

    @classmethod
    def mutate(cls, parent):
        child = cls(parent.num_tri, parent.size)
        child.position = parent.position.copy()
        child.colors = parent.colors.copy()
        child.position[:, :, :2] += np.random.rand(
            child.position.shape[0], child.position.shape[1], 2)*0.1
        child.colors += np.random.rand(
            child.colors.shape[0], child.colors.shape[1])*0.1
        return child

def main(inpName, out_shape=(256, 256),
    num_seed=100, num_tri=128, num_child=20,
    max_iter=10000, save_iter=100, save_path=None):
    # 声明全局变量
    img = cv2.imread(inpName)
    img = cv2.resize(img, out_shape)
    img = img.astype(np.float)/255.
    # 生成一系列的图片作为父本，选择其中最好的一个进行遗传
    parentList = []
    for i in range(num_seed):
        print('正在生成第%d个初代个体'%(i))
        parent = Canvas(size=out_shape)
        parent.calc_match_rate(img)
        parentList.append(parent)
    parent = sorted(parentList,key = lambda x:x.match_rate)[0]
    # 进入遗传算法的循环
    i = 0
    while i < max_iter:
        childList = []
        # 每一代从父代中变异出10个个体
        for j in range(num_child):
            child = Canvas.mutate(parent)
            child.calc_match_rate(img)
            childList.append(child)
        child = sorted(childList,key = lambda x:x.match_rate)[0]
        print ('%10d parent rate %11d \t child1 rate %.3f' % (i, parent.match_rate, child.match_rate))
        parent = parent if parent.match_rate < child.match_rate else child
        # 如果子代比父代更适应环境，那么子代成为新的父代
        # 否则保持原样
        child = None
        if i % save_iter == 0:
            # 每隔LOOP代保存一次图片
            rgb, count = parent.draw()
            out = (rgb*255).astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_path, '%06d.jpg'%(i)), out)
            print(parent.match_rate)
        i += 1
    
if __name__ == "__main__":
    print('开始进行遗传算法')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='data/head.jpg')
    parser.add_argument('-o', '--out', type=str, default='output')
    parser.add_argument('-l', '--loop', type=int, default=100)
    parser.add_argument('-n', '--num', type=int, default=128)

    args = parser.parse_args()
    import os
    save_path = args.out
    os.makedirs(save_path, exist_ok=True)
    main(args.input, num_tri=args.num, save_path=save_path)
