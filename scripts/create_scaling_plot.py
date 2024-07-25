# load hands-R.jpg and convert to pytorch tensor
import torch
from PIL import Image
import matplotlib.pyplot as plt
from src.domain import Domain
from src.interpolation import SplineInter
from src.plotting import view_image_2d, plot_grid_2d
torch.set_default_dtype(torch.float64)

R = Image.open('./data/hands-R.jpg')
m = 128
R = R.resize((m, m))
R = torch.fliplr(torch.tensor(R.getdata(), dtype=torch.float64).view(m,m).transpose(0,1))

domain = Domain(torch.tensor([0.0, 20.0 ,0.0, 25.0]), torch.tensor([m, m]))
xc = domain.getCellCenteredGrid().view(-1,2)

scaling = torch.tensor([1e4, 1e3, 1e2, 1e1, 1e0, 1e-3])

plt.figure(figsize=(12,3))
for i,theta in enumerate(scaling):
    Rimg = SplineInter(R, domain, 'moments', theta)
    Rc = Rimg(xc)
    plt.subplot(1,6,i+1)
    view_image_2d(Rc, domain)
    plt.title("$\\theta=${:.0e}".format(theta.item()), fontsize=14)

plt.tight_layout()
plt.savefig('./results/figs/scaling.png')
plt.show()