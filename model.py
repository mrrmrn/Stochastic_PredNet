import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class KyotoRadarDataset(Dataset):
    """Kyoto Radar dataset."""

    def __init__(self, root_dir, times, steps, tensor=False):
        """
        Args:
            root_dir (string)             : Directory with all the images.
            times (pandas datetimeindex)  : Dates of the radar images
            steps (int)                   : Number of steps in the sequence
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        self.times = times
        self.steps = steps
        self.tensor = tensor
    
    def __len__(self):
        return len(self.times) - self.steps

    def __getitem__(self, idx):
        sample = []
        
        t1 = self.times[idx]
        t0 = t1 - self.steps//2 * pd.Timedelta('5min')
        t2 = t1 + self.steps//2 * pd.Timedelta('5min')
        #print(t1)
        for t in pd.date_range(t0, t2, freq='5min')[:-1]:
            filename = "{:4d}{:02d}{:02d}{:02d}{:02d}.npz".format(t.year,t.month,t.day,t.hour,t.minute)
            res = read_prate(self.root_dir+filename)
            #img = res[::10,::10]/15.
            #img = img[68:68+96*9:9,128:128+96*9:9,:] # luca's sampling: 96 x 96
            #img = res[34:34+96*9:9,64:64+96*9:9]/15.
            #img = res[68:68+96*9:9,64:64+96*9:9]/15.
            img = res[68:68+96*9:9,96:96+96*9:9]/15.
            
            if self.tensor:
                img = np.expand_dims(img, 2)
                img = np.transpose(img, (2,0,1))
            
            sample.append(img)

        sample = np.array(sample)

        return sample.astype(np.float32)
    
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        #return h_next, c_next
        return h_next, (h_next, c_next)


    def reset_parameters(self):
        self.conv.reset_parameters()

    def init_hidden(self, batch_size, width, height):
        return (torch.zeros(batch_size, self.hidden_dim, width, height),
                torch.zeros(batch_size, self.hidden_dim, width, height))
        #return (torch.zeros(batch_size, self.hidden_dim, width, height).cuda(),
        #        torch.zeros(batch_size, self.hidden_dim, width, height).cuda())
        #return (torch.rand(batch_size, self.hidden_dim, width, height).cuda(),
        #        torch.rand(batch_size, self.hidden_dim, width, height).cuda())


class SatLU(nn.Module):

    def __init__(self, lower=0, upper=15/15, inplace=False):
        super(SatLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return F.hardtanh(input, self.lower, self.upper, self.inplace)


    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' ('\
            + 'min_val=' + str(self.lower) \
            + ', max_val=' + str(self.upper) \
            + inplace_str + ')'
class PredNet(nn.Module):
    def __init__(self, R_channels, A_channels):
        super(PredNet, self).__init__()
        self.r_channels = R_channels + (0, )  # for convenience
        self.a_channels = A_channels
        self.n_layers = len(R_channels)

        for i in range(self.n_layers):
            cell = ConvLSTMCell(2 * self.a_channels[i] + self.r_channels[i+1], 
                                self.r_channels[i], KERNEL_SIZE, True)

            setattr(self, 'cell{}'.format(i), cell)

        for i in range(self.n_layers):
            conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], KERNEL_SIZE, padding=PADDING), nn.ReLU())
            if i == 0:
                conv.add_module('satlu', SatLU())
            setattr(self, 'conv{}'.format(i), conv)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(nn.Conv2d(2* self.a_channels[l], self.a_channels[l+1], KERNEL_SIZE, padding=PADDING), self.maxpool)
            setattr(self, 'update_A{}'.format(l), update_A)

        self.reset_parameters()

    def reset_parameters(self):
        for l in range(self.n_layers):
            cell = getattr(self, 'cell{}'.format(l))
            cell.reset_parameters()


    def forward(self, input, time_steps=1, forecast_steps=0, mode='error'):
        
        R_seq = [None] * self.n_layers
        H_seq = [None] * self.n_layers
        E_seq = [None] * self.n_layers
        
        batch_size, input_steps, channels, width, height = input.size()
        
        w = width
        h = height
        for l in range(self.n_layers):
            E_seq[l] = torch.randn(batch_size, 2*self.a_channels[l], w, h)#.cuda()
            R_seq[l] = torch.randn(batch_size, self.r_channels[l], w, h)#.cuda()
            w = w//2
            h = h//2
        
        if mode=='error':
            total_error = []
        else:
            output = []
        
        for t in range(time_steps + forecast_steps):
            if t < input_steps:
                frame_input = input[:,t]
                frame_input = frame_input.type(torch.FloatTensor)
                #frame_input = frame_input.type(torch.cuda.FloatTensor)
            else:
                frame_input = None
            
            for l in reversed(range(self.n_layers)):
                cell = getattr(self, 'cell{}'.format(l))
                
                if t == 0:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = (R, R)
                else:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = H_seq[l]
                
                if l == self.n_layers - 1:
                    R, hx = cell(E, hx)
                else:
                    tmp = torch.cat((E, F.interpolate(R_seq[l+1], scale_factor=2)), 1)
                    R, hx = cell(tmp, hx)
                
                R_seq[l] = R
                H_seq[l] = hx
                
            for l in range(self.n_layers):
                conv = getattr(self, 'conv{}'.format(l))
                A_hat = conv(R_seq[l])
                
                if l == 0:
                    frame_prediction = A_hat
                    if t < time_steps:
                        A = frame_input
                    else:
                        A = frame_prediction
                
                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
                E = torch.cat([pos, neg],1)
                E_seq[l] = E
                
                if l < self.n_layers - 1:
                    update_A = getattr(self, 'update_A{}'.format(l))
                    A = update_A(E)
                
            if mode == 'error':
                error = torch.mean((frame_input - frame_prediction)**2) # MSE!
                total_error.append(error)
            else:
                output.append(frame_prediction)
            
        if mode == 'error':
            return torch.stack(total_error, 0)
        else:
            return torch.stack(output, 1)
        

# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE//2
BATCH_SIZE = 8

A_channels = (1, 16, 32, 64)
R_channels = (1, 16, 32, 64)

model = PredNet(R_channels, A_channels)
if torch.cuda.is_available():
    print('Using GPU.')
    model.cuda()
    

