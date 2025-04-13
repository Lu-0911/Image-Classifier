import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os

# 加载CIFAR-10数据集
def load_cifar10_data(data_dir):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    # 加载训练数据
    for i in range(1, 6):
        with open(os.path.join(data_dir, f'data_batch_{i}'), 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
            X_train.append(data_dict[b'data'])
            y_train.append(data_dict[b'labels'])
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    
    # # 通过翻转和增加训练数据
    # X_train = np.concatenate((X_train, np.fliplr(X_train)))
    # y_train = np.concatenate((y_train, y_train))
    
    # 加载测试数据
    with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        X_test = data_dict[b'data']
        y_test = data_dict[b'labels']
        
    # 数据预处理
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    # # 部分数据添加噪声
    # noise_mask = np.random.rand(X_train.shape[0]) < 0.5
    # X_train[noise_mask] += np.random.normal(0, 0.1, X_train[noise_mask].shape)
    
    # 将标签转换为one-hot编码
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    
    # 划分验证集
    n_val = int(X_train.shape[0] * 0.1)
    X_train, y_train = shuffle_data(X_train, y_train)
    X_val = X_train[:n_val]
    y_val = y_train[:n_val]
    X_train = X_train[n_val:]
    y_train = y_train[n_val:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# 数据打乱函数
def shuffle_data(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]

# 激活函数
def activation(X, activation):
    if activation == 'relu':
        return np.maximum(0, X)
    elif activation == 'leaky_relu':
        return np.maximum(0.01 * X, X)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-X))
    elif activation == 'tanh':
        return np.tanh(X)
    elif activation == 'linear':
        return X
    else:
        raise ValueError('Invalid activation function')

# 激活函数的导数
def activation_derivative(X, activation):
    if activation == 'relu':
        return np.where(X > 0, 1, 0)
    elif activation == 'leaky_relu':
        return np.where(X > 0, 1, 0.01)
    elif activation == 'sigmoid':
        sigmoid_x = 1 / (1 + np.exp(-X))
        return sigmoid_x * (1 - sigmoid_x)
    elif activation == 'tanh':
        return 1 - np.tanh(X) ** 2
    elif activation == 'linear':
        return np.ones_like(X)
    else:
        raise ValueError('Invalid activation function')

# 全连接层 (FCN)
class FullyConnectedLayer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        # He初始化
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))

    # 前向传播
    def forward(self, X):
        self.X = X
        self.z = np.dot(X, self.W) + self.b
        self.a = activation(self.z, self.activation)
        return self.a

    # 反向传播
    def backward(self, grad_output, learning_rate, lambda_):
        grad_a = grad_output * activation_derivative(self.z, self.activation)
        grad_W = np.dot(self.X.T, grad_a) / grad_a.shape[0]
        grad_b = np.mean(grad_a, axis=0, keepdims=True)
        grad_x = np.dot(grad_a, self.W.T)
        grad_W += lambda_ * self.W

        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b
        return grad_x

# 批量归一化层 (BN)
class BatchNormLayer:
    def __init__(self, input_size, momentum=0.9, eps=1e-8, mode='train'):
        self.momentum = momentum
        self.eps = eps
        self.mode = mode
        self.gamma = np.ones((1, input_size))
        self.beta = np.zeros((1, input_size))
        self.running_mean = np.zeros((1, input_size))
        self.running_var = np.ones((1, input_size))
        
    # 前向传播
    def forward(self, X):
        if self.mode == 'train':
            self.batch_mean = np.mean(X, axis=0, keepdims=True)
            self.batch_var = np.var(X, axis=0, keepdims=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        else:
            self.batch_mean = self.running_mean
            self.batch_var = self.running_var
        self.x_centered = X - self.batch_mean
        self.std = np.sqrt(self.batch_var + self.eps)
        self.x_norm = self.x_centered / self.std
        return self.gamma * self.x_norm + self.beta

    #反向传播
    def backward(self, grad_output, learning_rate):        
        grad_xnorm = grad_output * self.gamma
        grad_var = np.sum(grad_xnorm * self.x_centered * (-0.5) * (self.batch_var + self.eps)**(-1.5), axis=0)
        grad_mean = np.sum(grad_xnorm * (-1 / self.std), axis=0) + grad_var * np.mean(-2 * self.x_centered, axis=0)        
        grad_x = grad_xnorm / self.std + (grad_var * 2 * self.x_centered + grad_mean) / grad_output.shape[0]
        grad_gamma = np.sum(grad_output * self.x_norm, axis=0, keepdims=True)
        grad_beta = np.sum(grad_output, axis=0, keepdims=True)

        self.gamma -= learning_rate * grad_gamma
        self.beta -= learning_rate * grad_beta
        return grad_x

# # Dropout层（性能提升不明显，未使用）
# class DropoutLayer:
#     def __init__(self, p=0.5, mode='train'):
#         self.p = p
#         self.mode = mode

#     def forward(self, X):
#         if self.mode == 'train':
#             self.mask = (np.random.rand(*X.shape) < self.p) / self.p
#             return X * self.mask
#         return X

#     def backward(self, grad_output):
#         if self.mode == 'train':
#             return grad_output * self.mask
#         return grad_output

# 三层神经网络
# 网络结构：输入 -> 全连接层 -> BN层 -> 全连接层 -> BN层 -> 全连接层 -> 输出
class ThreeLayerNet:
    # 初始化，调用FCN类和BN类
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, activation='relu'):
        self.fc1 = FullyConnectedLayer(input_size, hidden_size1, activation)
        self.bn1 = BatchNormLayer(hidden_size1)
        self.fc2 = FullyConnectedLayer(hidden_size1, hidden_size2, activation)
        self.bn2 = BatchNormLayer(hidden_size2)
        self.fc3 = FullyConnectedLayer(hidden_size2, output_size, 'linear')
        self.layers = [self.fc1, self.bn1, self.fc2, self.bn2, self.fc3]
        # self.layers = [self.fc1, self.fc2, self.fc3]
        self.activation = activation
        self.mode = 'train'
        
    # 前向传播
    def forward(self, X, mode='train'):
        self.set_mode(mode)
        for layer in self.layers:
            X = layer.forward(X)
        # 输出层使用softmax函数
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)
        
    # 反向传播以及参数更新
    def backward(self, grad, learning_rate, lambda_):
        for layer in reversed(self.layers):
            if isinstance(layer, FullyConnectedLayer):
                grad = layer.backward(grad, learning_rate, lambda_)
            elif isinstance(layer, BatchNormLayer):
                grad = layer.backward(grad, learning_rate)
    # 设置模式
    def set_mode(self, mode):
        for layer in self.layers:
            if isinstance(layer, (BatchNormLayer)):
                layer.mode = mode

# 模型深拷贝
def model_copy(model):
    new_model = ThreeLayerNet(model.fc1.input_size, model.fc1.output_size, model.fc2.output_size,
        model.fc3.output_size, model.activation)
    new_model.fc1.W = np.copy(model.fc1.W)
    new_model.fc1.b = np.copy(model.fc1.b)
    new_model.fc2.W = np.copy(model.fc2.W)
    new_model.fc2.b = np.copy(model.fc2.b)
    new_model.fc3.W = np.copy(model.fc3.W)
    new_model.fc3.b = np.copy(model.fc3.b)
    new_model.bn1.gamma = np.copy(model.bn1.gamma)
    new_model.bn1.beta = np.copy(model.bn1.beta)
    new_model.bn1.running_mean = np.copy(model.bn1.running_mean)
    new_model.bn1.running_var = np.copy(model.bn1.running_var)
    new_model.bn2.gamma = np.copy(model.bn2.gamma)
    new_model.bn2.beta = np.copy(model.bn2.beta)
    new_model.bn2.running_mean = np.copy(model.bn2.running_mean)
    new_model.bn2.running_var = np.copy(model.bn2.running_var)
    return new_model

# 训练函数
def train(model, X_train, y_train, X_val, y_val,
         learning_rate, lambda_, decay_rate=0.90, batch_size=128, epochs=100):
    
    n_samples = X_train.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))
    train_losses = []
    val_losses = []
    accuracies = []
    best_accuracy = 0
    best_model = None
    patience = 10
    no_improve_count = 0
    
    for epoch in range(epochs):
        train_loss = 0
        X_train, y_train = shuffle_data(X_train, y_train)
        # 小批次训练
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            # 前向传播
            y_hat = model.forward(X_batch, 'train')
            
            # 交叉熵损失与L2正则化
            batch_loss = -np.mean(y_batch * np.log(y_hat + 1e-8)) 
            batch_loss += lambda_/ (2 * batch_size) * (np.sum(model.fc1.W ** 2) + 
                 np.sum(model.fc2.W ** 2) + np.sum(model.fc3.W ** 2))
            train_loss += batch_loss * (end_idx - start_idx) / n_samples
            # 反向传播以及参数更新
            grad = y_hat - y_batch
            model.backward(grad, learning_rate, lambda_)    
        train_losses.append(train_loss)
        
        # 验证和早停机制
        val_loss = -np.mean(y_val * np.log(model.forward(X_val, 'test') + 1e-8))
        val_losses.append(val_loss)
        val_accuracy = test(model, X_val, y_val)
        accuracies.append(val_accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model_copy(model)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print('Early stopping triggered.')
                break
                
        # 学习率衰减（策略可选）
        # 指数衰减
        learning_rate *= decay_rate
        #阶梯衰减
        # if epoch < 5:
        #     learning_rate = 0.01
        # elif epoch < 10:
        #     learning_rate = 0.005
        # elif epoch < 15:
        #     learning_rate = 0.001
        # else:
        #     learning_rate = 0.0001
        # 余弦退火
        # learning_rate = 0.01 * (1 + np.cos(np.pi * epoch / epochs)) / 2
        
        # 打印训练信息
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}, Loss: {train_loss:.4f}, Acc: {val_accuracy:.4f}')
    # 绘制损失曲线和准确率曲线
    plt.plot(train_losses)
    plt.ylabel('Tran_losses')
    plt.xlabel('Epining Loss')
    plt.title('Training Loss Curve')
    plt.savefig('D:/py/train_loss.png')
    plt.close()
    plt.plot(val_losses)
    plt.ylabel('Val_losses')
    plt.xlabel('Epining Loss')
    plt.title('Validation Loss Curve')
    plt.savefig('D:/py/val_loss.png')
    plt.close()
    plt.plot(accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.savefig('D:/py/val_acc.png')
    plt.close()
             
    # 保存最优模型
    with open('D:/py/model.pkl', 'wb') as f:
        pickle.dump(best_model, f)  
    return best_model

# 测试函数
def test(model, X_test, y_test):
    model.set_mode('test')
    y_hat = model.forward(X_test)
    accuracy = np.mean(np.argmax(y_hat, axis=1) == np.argmax(y_test, axis=1))
    return accuracy

# 评估参数组合
def evaluate_params(params, X_train, y_train, X_val, y_val):
    hs1, hs2, lr, lmbda, bs = params
    start_time = time.time()
    # 训练和测试模型
    model = ThreeLayerNet(3072, hs1, hs2, 10, 'leaky_relu')
    model = train(model, X_train, y_train, X_val, y_val,
                 learning_rate=lr, lambda_=lmbda, batch_size=bs, epochs=10)
    accuracy = test(model, X_val, y_val)
    time_cost = time.time() - start_time
    print(f'hs1: {hs1}, hs2: {hs2}, lr: {lr}, lmbda: {lmbda}, bs: {bs}, accuracy: {accuracy}, time_cost: {time_cost}')
    return (params, accuracy, time_cost)

# 参数查找
from itertools import product
from multiprocessing import Pool
from functools import partial

def parameter_search(X_train, y_train, X_val, y_val):
    best_accuracy = 0
    best_params = {}
    # 参数搜索空间
    param_grid = {
        'hidden_size1': [128, 256],
        'hidden_size2': [64, 128, 256],
        'learning_rate': [0.001, 0.005, 0.01],
        'lambda_': [0.0001, 0.001, 0.01],
        'batch_size': [64, 128, 256]
        }
    param_combinations = list(product(*param_grid.values()))

    result_file = 'D:/py/parameter_search_results.csv'
    with open(result_file, 'w') as f:
        f.write(','.join(param_grid.keys()) + ',accuracy,time_cost\n')

    # 并行处理
    with Pool(processes=4) as pool:
        results = []
        for params in param_combinations:
            func = partial(evaluate_params, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
            results.append(pool.apply_async(func, (params,)))
        
        # 收集并处理结果
        for res in results:
            params, accuracy, time_cost = res.get()
            with open(result_file, 'a') as f:
                f.write(f'{params[0]},{params[1]},{params[2]},{params[3]},{params[4]},{accuracy:.4f},{time_cost:.1f}\n')
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = dict(zip(param_grid.keys(), params))
    print(f'\nBest Accuracy: {best_accuracy:.4f}')
    print('Best Parameters:', best_params)
    return best_params
    
# 模型参数可视化
def visualize_model_parameters(model_path):
    # 加载训练好的模型
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    plt.figure(figsize=(15, 10))
    
    # 可视化全连接层权重分布
    plt.subplot(2, 3, 1)
    plt.hist(model.fc1.W.flatten(), bins=50, alpha=0.7, label='FC1 Weights')
    plt.hist(model.fc2.W.flatten(), bins=50, alpha=0.7, label='FC2 Weights')
    plt.hist(model.fc3.W.flatten(), bins=50, alpha=0.7, label='FC3 Weights')
    plt.title('Weight Distribution')
    plt.legend()
    
    # 可视化BN层参数
    plt.subplot(2, 3, 2)
    plt.scatter(model.bn1.gamma.flatten(), model.bn1.beta.flatten(), alpha=0.6)
    plt.title('BN1 Gamma-Beta Distribution')
    plt.xlabel('Gamma'), plt.ylabel('Beta')
    
    plt.subplot(2, 3, 3)
    plt.scatter(model.bn2.gamma.flatten(), model.bn2.beta.flatten(), alpha=0.6)
    plt.title('BN2 Gamma-Beta Distribution')
    plt.xlabel('Gamma'), plt.ylabel('Beta')
    
    # 权重热力图（取部分神经元）
    plt.subplot(2, 3, 4)
    plt.imshow(model.fc1.W[:50, :50], cmap='coolwarm', aspect='auto')
    plt.title('FC1 Weight Heatmap (Partial)')
    plt.colorbar()
    
    plt.subplot(2, 3, 5)
    plt.imshow(model.fc2.W[:50, :50], cmap='coolwarm', aspect='auto')
    plt.title('FC2 Weight Heatmap (Partial)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('D:/py/model_parameters.png')
    plt.close()


def main():
    data_dir = 'path_to_data'
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_cifar10_data(path_to_data)

    # # 超参数查找
    # best_params = parameter_search(X_train, y_train, X_val, y_val)
    # print('Best Parameters:', best_params)

    # 训练模型
    model = ThreeLayerNet(3072, 256, 64, 10, 'leaky_relu')
    start_time = time.time()
    model = train(model, X_train, y_train, X_val, y_val,
                 learning_rate=0.01, lambda_=0.001, batch_size=128, epochs=100)
    time_cost = time.time() - start_time
    print(f'Training Time: {time_cost:.1f} seconds')
    
    # 测试模型
    test_accuracy = test(model, X_test, y_test)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    #模型可视化
    visualize_model_parameters('D:/py/model.pkl')

if __name__ == '__main__':
    main()
