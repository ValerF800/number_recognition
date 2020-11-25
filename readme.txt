


#install loss ctc for Recurrent network

!git clone https://github.com/SeanNaren/warp-ctc.git
os.chdir('/content/warp-ctc')
!mkdir build
os.chdir('build')
!cmake ..
!make
os.chdir('/content/warp-ctc/pytorch_binding')
!python setup.py install
!cp /usr/local/lib/python3.6/dist-packages/warpctc_pytorch-0.1-py3.6-linux-x86_64.egg/warpctc_pytorch/_warp_ctc.cpython-36m-x86_64-linux-gnu.so /content/warp-ctc/pytorch_binding/warpctc_pytorch/

os.chdir('/content')
!git clone https://github.com/meijieru/crnn.pytorch

sys.path.append('/content/crnn.pytorch')
sys.path.append('/content/warp-ctc/pytorch_binding')
