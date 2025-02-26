import torch
import torch.nn as nn
import math 


def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)

#CRNN
class Crnnplus(nn.Module):
    def __init__(self, num_freq, num_class):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     num_class: int, the number of output classes
        ##############################
        super(Crnnplus, self).__init__()
        self.inputdim = num_freq
        self.outputdim = num_class


        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)


        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)
        
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)
        
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)

        self.bigru = nn.GRU(input_size = 64 * 8, hidden_size = 128, num_layers = 1, batch_first = True, bidirectional = True)
        self.fc = nn.Linear(256, num_class, bias=True)
        
        self.relu = nn.ReLU()

        self.init_bn(self.bn0)
        self.init_bn(self.bn1)
        self.init_bn(self.bn2)
        self.init_bn(self.bn3)
        self.init_layer(self.fc)
        self.init_gru(self.bigru)

    def init_bn(self, bn):
        bn.bias.data.fill_(0.)
        bn.weight.data.fill_(1.)
    
    def init_layer(self, layer):
        nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)
            
    def init_gru(self, rnn):
        def _concat_init(tensor, init_funcs):
            (length, fan_out) = tensor.shape
            fan_in = length // len(init_funcs)
        
            for (i, init_func) in enumerate(init_funcs):
                init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
            
        def _inner_uniform(tensor):
            fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
            nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
        
        for i in range(rnn.num_layers):
            _concat_init(
                getattr(rnn, 'weight_ih_l{}'.format(i)),
                [_inner_uniform, _inner_uniform, _inner_uniform]
            )
            torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

            _concat_init(
                getattr(rnn, 'weight_hh_l{}'.format(i)),
                [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
            )
            torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)

        

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_prob: [batch_size, time_steps, num_class]
        ##############################
        batch_size, time_steps, num_freq = x.shape[0], x.shape[1], x.shape[2]
        x = x.unsqueeze(1)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))
        
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, time_steps, -1)
        x, _ = self.bigru(x)
        x = torch.sigmoid(self.fc(x))
        
        return x
        

    def forward(self, x):
        frame_prob = self.detection(x)  # (batch_size, time_steps, class_num)
        clip_prob = linear_softmax_pooling(frame_prob)  # (batch_size, class_num)
        '''(samples_num, feature_maps)'''
        return {
            'clip_prob': clip_prob, 
            'frame_prob': frame_prob
        }

class Crnn(nn.Module):
    def __init__(self, num_freq, num_class):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     num_class: int, the number of output classes
        ##############################    
        super(Crnn, self).__init__()
        self.inputdim = num_freq
        self.outputdim = num_class


        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)


        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)
        
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)
        

        self.bigru = nn.GRU(input_size = 64 * 8, hidden_size = 128, num_layers = 1, batch_first = True, bidirectional = True)
        self.fc = nn.Linear(256, num_class, bias=True)
        
        self.relu = nn.ReLU()

        self.init_bn(self.bn0)
        self.init_bn(self.bn1)
        self.init_bn(self.bn2)
        self.init_layer(self.fc)
        self.init_gru(self.bigru)

    def init_bn(self, bn):
        bn.bias.data.fill_(0.)
        bn.weight.data.fill_(1.)
    
    def init_layer(self, layer):
        nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)
            
    def init_gru(self, rnn):
        def _concat_init(tensor, init_funcs):
            (length, fan_out) = tensor.shape
            fan_in = length // len(init_funcs)
        
            for (i, init_func) in enumerate(init_funcs):
                init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
            
        def _inner_uniform(tensor):
            fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
            nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
        
        for i in range(rnn.num_layers):
            _concat_init(
                getattr(rnn, 'weight_ih_l{}'.format(i)),
                [_inner_uniform, _inner_uniform, _inner_uniform]
            )
            torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

            _concat_init(
                getattr(rnn, 'weight_hh_l{}'.format(i)),
                [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
            )
            torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_prob: [batch_size, time_steps, num_class]
        ##############################
        batch_size, time_steps, num_freq = x.shape[0], x.shape[1], x.shape[2]
        x = x.unsqueeze(1)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))
        
        
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, time_steps, -1)
        x, _ = self.bigru(x)
        x = torch.sigmoid(self.fc(x))
        
        return x

    def forward(self, x):
        frame_prob = self.detection(x)  # (batch_size, time_steps, class_num)
        clip_prob = linear_softmax_pooling(frame_prob)  # (batch_size, class_num)
        '''(samples_num, feature_maps)'''
        return {
            'clip_prob': clip_prob, 
            'frame_prob': frame_prob
        }
    
class Crnn1(nn.Module):
    def __init__(self, num_freq, num_class):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     num_class: int, the number of output classes
        ##############################    
        super(Crnn1, self).__init__()
        self.num_freq = num_freq
        self.num_class = num_class


        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)


        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)
        
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)
        
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)
                    
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 128,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)

        self.bigru = nn.GRU(input_size = 64 * 8, hidden_size = 128, num_layers = 1, batch_first = True, bidirectional = True)
        self.fc = nn.Linear(256, num_class, bias=True)
        
        self.relu = nn.ReLU()

        self.init_bn(self.bn0)
        self.init_bn(self.bn1)
        self.init_bn(self.bn2)
        self.init_bn(self.bn3)
        self.init_bn(self.bn4)
        self.init_layer(self.fc)
        self.init_gru(self.bigru)

    def init_bn(self, bn):
        bn.bias.data.fill_(0.)
        bn.weight.data.fill_(1.)
    
    def init_layer(self, layer):
        nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)
            
    def init_gru(self, rnn):
        def _concat_init(tensor, init_funcs):
            (length, fan_out) = tensor.shape
            fan_in = length // len(init_funcs)
        
            for (i, init_func) in enumerate(init_funcs):
                init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
            
        def _inner_uniform(tensor):
            fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
            nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
        
        for i in range(rnn.num_layers):
            _concat_init(
                getattr(rnn, 'weight_ih_l{}'.format(i)),
                [_inner_uniform, _inner_uniform, _inner_uniform]
            )
            torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

            _concat_init(
                getattr(rnn, 'weight_hh_l{}'.format(i)),
                [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
            )
            torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_prob: [batch_size, time_steps, num_class]
        ##############################
        batch_size, time_steps, num_freq = x.shape[0], x.shape[1], x.shape[2]
        x = x.unsqueeze(1)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))

        x = self.relu(self.bn4(self.conv4(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))
        
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, time_steps, -1)
        x, _ = self.bigru(x)
        x = torch.sigmoid(self.fc(x))
        
        return x

    def forward(self, x):
        frame_prob = self.detection(x)  # (batch_size, time_steps, num_class)
        clip_prob = linear_softmax_pooling(frame_prob)  # (batch_size, num_class)
        '''(samples_num, feature_maps)'''
        return {
            'clip_prob': clip_prob, 
            'frame_prob': frame_prob
        }
    
class Crnn2(nn.Module):
    def __init__(self, num_freq, num_class):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     num_class: int, the number of output classes
        ##############################    
        super(Crnn2, self).__init__()
        self.num_freq = num_freq
        self.num_class = num_class


        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)
        
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)
        
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)
                    
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 128,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)

        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 256,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)

        self.bigru = nn.GRU(input_size = 64 * 8, hidden_size = 128, num_layers = 1, batch_first = True, bidirectional = True)
        self.fc = nn.Linear(256, num_class , bias=True)
        
        self.relu = nn.ReLU()

        self.init_bn(self.bn0)
        self.init_bn(self.bn1)
        self.init_bn(self.bn2)
        self.init_bn(self.bn3)
        self.init_bn(self.bn4)
        self.init_bn(self.bn5)
        self.init_layer(self.fc)
        self.init_gru(self.bigru)

    def init_bn(self, bn):
        bn.bias.data.fill_(0.)
        bn.weight.data.fill_(1.)
    
    def init_layer(self, layer):
        nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)
            
    def init_gru(self, rnn):
        def _concat_init(tensor, init_funcs):
            (length, fan_out) = tensor.shape
            fan_in = length // len(init_funcs)
        
            for (i, init_func) in enumerate(init_funcs):
                init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
            
        def _inner_uniform(tensor):
            fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
            nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
        
        for i in range(rnn.num_layers):
            _concat_init(
                getattr(rnn, 'weight_ih_l{}'.format(i)),
                [_inner_uniform, _inner_uniform, _inner_uniform]
            )
            torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

            _concat_init(
                getattr(rnn, 'weight_hh_l{}'.format(i)),
                [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
            )
            torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_prob: [batch_size, time_steps, num_class]
        ##############################
        batch_size, time_steps, num_freq = x.shape[0], x.shape[1], x.shape[2]
        x = x.unsqueeze(1)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))

        x = self.relu(self.bn4(self.conv4(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))

        x = self.relu(self.bn5(self.conv5(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))
        
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, time_steps, -1)
        x, _ = self.bigru(x)
        x = torch.sigmoid(self.fc(x))
        
        return x

    def forward(self, x):
        frame_prob = self.detection(x)  # (batch_size, time_steps, num_class)
        clip_prob = linear_softmax_pooling(frame_prob)  # (batch_size, num_class)
        '''(samples_num, feature_maps)'''
        return {
            'clip_prob': clip_prob, 
            'frame_prob': frame_prob
        }


class Crnn3(nn.Module):
    def __init__(self, num_freq, num_class):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     num_class: int, the number of output classes
        ##############################    
        super(Crnn3, self).__init__()
        self.num_freq = num_freq
        self.num_class = num_class


        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)
        
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)
        
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)
                    
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 128,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)

        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 256,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)

        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 512,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding = (1, 1), bias=False)

        self.bigru = nn.GRU(input_size = 64 * 8, hidden_size = 128, num_layers = 1, batch_first = True, bidirectional = True)
        self.fc = nn.Linear(256, num_class, bias=True)
        
        self.relu = nn.ReLU()

        self.init_bn(self.bn0)
        self.init_bn(self.bn1)
        self.init_bn(self.bn2)
        self.init_bn(self.bn3)
        self.init_bn(self.bn4)
        self.init_bn(self.bn5)
        self.init_bn(self.bn6)
        self.init_layer(self.fc)
        self.init_gru(self.bigru)

    def init_bn(self, bn):
        bn.bias.data.fill_(0.)
        bn.weight.data.fill_(1.)
    
    def init_layer(self, layer):
        nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)
            
    def init_gru(self, rnn):
        def _concat_init(tensor, init_funcs):
            (length, fan_out) = tensor.shape
            fan_in = length // len(init_funcs)
        
            for (i, init_func) in enumerate(init_funcs):
                init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
            
        def _inner_uniform(tensor):
            fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
            nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
        
        for i in range(rnn.num_layers):
            _concat_init(
                getattr(rnn, 'weight_ih_l{}'.format(i)),
                [_inner_uniform, _inner_uniform, _inner_uniform]
            )
            torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

            _concat_init(
                getattr(rnn, 'weight_hh_l{}'.format(i)),
                [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
            )
            torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_prob: [batch_size, time_steps, num_class]
        ##############################
        batch_size, time_steps, num_freq = x.shape[0], x.shape[1], x.shape[2]
        x = x.unsqueeze(1)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))

        x = self.relu(self.bn4(self.conv4(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))

        x = self.relu(self.bn5(self.conv5(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))

        x = self.relu(self.bn6(self.conv6(x)))
        x = nn.functional.max_pool2d(x, kernel_size = (1, 2), stride = (1, 2))
        
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, time_steps, -1)
        x, _ = self.bigru(x)
        x = torch.sigmoid(self.fc(x))
        
        return x

    def forward(self, x):
        frame_prob = self.detection(x)  # (batch_size, time_steps, num_class)
        clip_prob = linear_softmax_pooling(frame_prob)  # (batch_size, num_class)
        '''(samples_num, feature_maps)'''
        return {
            'clip_prob': clip_prob, 
            'frame_prob': frame_prob
        }