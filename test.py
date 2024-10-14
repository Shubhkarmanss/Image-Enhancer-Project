# import os.path as osp
# import glob
# import cv2
# import numpy as np
# import torch
# import RRDBNet_arch as arch
# import os


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
# device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# # device = torch.device('cpu')

# test_img_folder = 'LR/*'

# model = arch.RRDBNet(3, 3, 64, 23, gc=32)
# model.load_state_dict(torch.load(model_path), strict=True)
# model.eval()
# model = model.to(device)

# print('Model path {:s}. \nTesting...'.format(model_path))

# idx = 0
# for path in glob.glob(test_img_folder):
#     idx += 1
#     base = osp.splitext(osp.basename(path))[0]
#     print(idx, base)
#     # read images
#     img = cv2.imread(path, cv2.IMREAD_COLOR)
#     img = img * 1.0 / 255
#     img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#     img_LR = img.unsqueeze(0)
#     img_LR = img_LR.to(device)

#     with torch.no_grad():
#         output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
#     output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
#     output = (output * 255.0).round()
#     cv2.imwrite('results/{:s}_rlt.png'.format(base), output)

# import os.path as osp
# import glob
# import cv2
# import numpy as np
# import torch
# import RRDBNet_arch as arch
# import os
# import streamlit as st

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
# device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# # device = torch.device('cpu')

# model = arch.RRDBNet(3, 3, 64, 23, gc=32)
# model.load_state_dict(torch.load(model_path), strict=True)
# model.eval()
# model = model.to(device)

# st.title('Image Super-Resolution Demo')

# test_img_folder = st.file_uploader('Upload test images', type=['png', 'jpg'], accept_multiple_files=True)

# if test_img_folder is not None:
#     st.write('Testing...')
#     idx = 0
#     for path in test_img_folder:
#         idx += 1
#         base = osp.splitext(osp.basename(path.name))[0]
#         st.write(idx, base)
#         # read images
#         img = cv2.imdecode(np.fromstring(path.read(), np.uint8), cv2.IMREAD_COLOR)
#         img = img * 1.0 / 255
#         img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#         img_LR = img.unsqueeze(0)
#         img_LR = img_LR.to(device)

#         with torch.no_grad():
#             output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
#         output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
#         output = (output * 255.0).round()
#         cv2.imwrite('results/{:s}_rlt.png'.format(base), output)

#         st.image(output.astype(np.uint8), caption=base, use_column_width=True)


# import os.path as osp
# import glob
# import cv2
# import numpy as np
# import torch
# import RRDBNet_arch as arch
# import os
# import streamlit as st

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
# device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# # device = torch.device('cpu')

# model = arch.RRDBNet(3, 3, 64, 23, gc=32)
# model.load_state_dict(torch.load(model_path), strict=True)
# model.eval()
# model = model.to(device)

# st.title('Image Super-Resolution and Editor')

# test_img_folder = st.file_uploader('Upload test images', type=['png', 'jpg'], accept_multiple_files=True)

# if test_img_folder is not None:
#     st.write('Testing...')
#     idx = 0
#     for path in test_img_folder:
#         idx += 1
#         base = osp.splitext(osp.basename(path.name))[0]
#         st.write(idx, base)
#         # read images
#         img = cv2.imdecode(np.fromstring(path.read(), np.uint8), cv2.IMREAD_COLOR)
#         img = img * 1.0 / 255
#         img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#         img_LR = img.unsqueeze(0)
#         img_LR = img_LR.to(device)

#         with torch.no_grad():
#             output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
#         output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
#         output = (output * 255.0).round()
#         cv2.imwrite('results/{:s}_rlt.png'.format(base), output)

#         st.image(output.astype(np.uint8), caption=base, use_column_width=True)

#         # Add photo editor functionality
#         st.write('Edit the image:')
#         brightness = st.slider('Brightness', -100, 100, 0)
#         contrast = st.slider('Contrast', -100, 100, 0)
#         gamma = st.slider('Gamma', 0.1, 10.0, 1.0, 0.1)
#         img_edit = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR)
#         img_edit = cv2.convertScaleAbs(img_edit, alpha=gamma, beta=brightness)
#         img_edit = cv2.addWeighted(img_edit, 1.0 + contrast / 100.0, img_edit, 0, 0)
#         st.image(cv2.cvtColor(img_edit, cv2.COLOR_BGR2RGB), caption='Edited Image', use_column_width=True)


# import os.path as osp
# import glob
# import cv2
# import numpy as np
# import torch
# import RRDBNet_arch as arch
# import os
# import streamlit as st

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
# device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# # device = torch.device('cpu')

# model = arch.RRDBNet(3, 3, 64, 23, gc=32)
# model.load_state_dict(torch.load(model_path), strict=True)
# model.eval()
# model = model.to(device)

# st.title('Image Super-Resolution and Editor')

# test_img_folder = st.file_uploader('Upload test images', type=['png', 'jpg'], accept_multiple_files=True)

# if test_img_folder is not None:
#     st.write('Testing...')
#     idx = 0
#     for path in test_img_folder:
#         idx += 1
#         base = osp.splitext(osp.basename(path.name))[0]
#         st.write(idx, base)
#         # read images
#         img = cv2.imdecode(np.fromstring(path.read(), np.uint8), cv2.IMREAD_COLOR)
#         img = img * 1.0 / 255
#         img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#         img_LR = img.unsqueeze(0)
#         img_LR = img_LR.to(device)

#         with torch.no_grad():
#             output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
#         output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
#         output = (output * 255.0).round()
#         cv2.imwrite('results/{:s}_rlt.png'.format(base), output)

#         st.image(output.astype(np.uint8), caption=base, use_column_width=True)

#         # Add photo editor functionality
#         st.write('Edit the image:')
#         brightness = st.slider('Brightness', -100, 100, 0)
#         contrast = st.slider('Contrast', -100, 100, 0)
#         gamma = st.slider('Gamma', 0.1, 10.0, 1.0, 0.1)
#         blur = st.slider('Blur', 0, 10, 0)
#         img_edit = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR)
#         img_edit = cv2.convertScaleAbs(img_edit, alpha=gamma, beta=brightness)
#         img_edit = cv2.addWeighted(img_edit, 1.0 + contrast / 100.0, img_edit, 0, 0)
#         img_edit = cv2.GaussianBlur(img_edit, (blur * 2 + 1, blur * 2 + 1), 0)
#         st.image(cv2.cvtColor(img_edit, cv2.COLOR_BGR2RGB), caption='Edited Image', use_column_width=True)

# import os.path as osp
# import glob
# import cv2
# import numpy as np
# import torch
# import RRDBNet_arch as arch
# import os
# import streamlit as st

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
# device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# # device = torch.device('cpu')

# model = arch.RRDBNet(3, 3, 64, 23, gc=32)
# model.load_state_dict(torch.load(model_path), strict=True)
# model.eval()
# model = model.to(device)

# st.title('Image Super-Resolution and Editor')

# test_img_folder = st.file_uploader('Upload test images', type=['png', 'jpg'], accept_multiple_files=True)

# if test_img_folder is not None:
#     st.write('Testing...')
#     idx = 0
#     for path in test_img_folder:
#         idx += 1
#         base = osp.splitext(osp.basename(path.name))[0]
#         st.write(idx, base)
#         # read images
#         img = cv2.imdecode(np.fromstring(path.read(), np.uint8), cv2.IMREAD_COLOR)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # convert to BGR format
#         img = img * 1.0 / 255
#         img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#         img_LR = img.unsqueeze(0)
#         img_LR = img_LR.to(device)

#         with torch.no_grad():
#             output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
#         output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
#         output = (output * 255.0).round()
#         cv2.imwrite('results/{:s}_rlt.png'.format(base), output)

#         st.image(output.astype(np.uint8), caption=base, use_column_width=True)

#         # Add photo editor functionality
#         st.write('Edit the image:')
#         brightness = st.slider('Brightness', -100, 100, 0)
#         contrast = st.slider('Contrast', -100, 100, 0)
#         gamma = st.slider('Gamma', 0.1, 10.0, 1.0, 0.1)
#         blur = st.slider('Blur', 0, 10, 0)
#         img_edit = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR)
#         img_edit = cv2.convertScaleAbs(img_edit, alpha=gamma, beta=brightness)
#         img_edit = cv2.addWeighted(img_edit, 1.0 + contrast / 100.0, img_edit, 0, 0)
#         img_edit = cv2.GaussianBlur(img_edit, (blur * 2 + 1, blur * 2 + 1), 0)
#         img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2RGB)
#         st.image(img_edit, caption='Edited Image', use_column_width=True)

# import os.path as osp
# import glob
# import cv2
# import numpy as np
# import torch
# import RRDBNet_arch as arch
# import os
# import streamlit as st

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
# device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# # device = torch.device('cpu')

# model = arch.RRDBNet(3, 3, 64, 23, gc=32)
# model.load_state_dict(torch.load(model_path), strict=True)
# model.eval()
# model = model.to(device)

# st.title('Image Super-Resolution and Editor')

# test_img_folder = st.file_uploader('Upload test images', type=['png', 'jpg'], accept_multiple_files=True)

# if test_img_folder is not None:
#     st.write('Testing...')
#     idx = 0
#     for path in test_img_folder:
#         idx += 1
#         base = osp.splitext(osp.basename(path.name))[0]
#         st.write(idx, base)
#         # read images
#         img = cv2.imdecode(np.fromstring(path.read(), np.uint8), cv2.IMREAD_COLOR)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # convert to BGR format
#         img = img * 1.0 / 255
#         img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#         img_LR = img.unsqueeze(0)
#         img_LR = img_LR.to(device)

#         with torch.no_grad():
#             output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
#         output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
#         output = (output * 255.0).round()
#         cv2.imwrite('results/{:s}_rlt.png'.format(base), output)

#         st.image(output.astype(np.uint8), caption=base, use_column_width=True)

#         # Add photo editor functionality
#         st.write('Edit the image:')
#         brightness = st.slider('Brightness', -100, 100, 0)
#         contrast = st.slider('Contrast', -100, 100, 0)
#         gamma = st.slider('Gamma', 0.1, 10.0, 1.0, 0.1)
#         blur = st.slider('Blur', 0, 10, 0)
#         filter_type = st.selectbox('Filter', ['None', 'Grayscale', 'Sepia', 'Invert'])
#         img_edit = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR)
#         img_edit = cv2.convertScaleAbs(img_edit, alpha=gamma, beta=brightness)
#         img_edit = cv2.addWeighted(img_edit, 1.0 + contrast / 100.0, img_edit, 0, 0)
#         img_edit = cv2.GaussianBlur(img_edit, (blur * 2 + 1, blur * 2 + 1), 0)
#         if filter_type == 'Grayscale':
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_GRAY2BGR)
#         elif filter_type == 'Sepia':
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2RGB)
#             img_edit = cv2.transform(img_edit, np.array([[0.393, 0.769, 0.189],
#                                                          [0.349, 0.686, 0.168],
#                                                          [0.272, 0.534, 0.131]]))
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_RGB2BGR)
#         elif filter_type == 'Invert':
#             img_edit = cv2.bitwise_not(img_edit)
#         img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2RGB)
#         st.image(img_edit, caption='Edited Image', use_column_width=True)

# import os.path as osp
# import glob
# import cv2
# import numpy as np
# import torch
# import RRDBNet_arch as arch
# import os
# import streamlit as st

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
# device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# # device = torch.device('cpu')

# model = arch.RRDBNet(3, 3, 64, 23, gc=32)
# model.load_state_dict(torch.load(model_path), strict=True)
# model.eval()
# model = model.to(device)

# st.title('Image Super-Resolution and Editor')

# test_img_folder = st.file_uploader('Upload test images', type=['png', 'jpg'], accept_multiple_files=True)

# if test_img_folder is not None:
#     st.write('Testing...')
#     idx = 0
#     for path in test_img_folder:
#         idx += 1
#         base = osp.splitext(osp.basename(path.name))[0]
#         st.write(idx, base)
#         # read images
#         img = cv2.imdecode(np.fromstring(path.read(), np.uint8), cv2.IMREAD_COLOR)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # convert to BGR format
#         img = img * 1.0 / 255
#         img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#         img_LR = img.unsqueeze(0)
#         img_LR = img_LR.to(device)

#         with torch.no_grad():
#             output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
#         output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
#         output = (output * 255.0).round()
#         cv2.imwrite('results/{:s}_rlt.png'.format(base), output)

#         st.image(output.astype(np.uint8), caption=base, use_column_width=True)

#         # Add photo editor functionality
#         st.write('Edit the image:')
#         brightness = st.slider('Brightness', -100, 100, 0)
#         contrast = st.slider('Contrast', -100, 100, 0)
#         gamma = st.slider('Gamma', 0.1, 10.0, 1.0, 0.1)
#         blur = st.slider('Blur', 0, 10, 0)
#         filter_type = st.selectbox('Filter', ['None', 'Grayscale', 'Sepia', 'Invert'])
#         flip_horizontal = st.checkbox('Flip Horizontal')
#         flip_vertical = st.checkbox('Flip Vertical')
#         rotate = st.slider('Rotate', -180, 180, 0)
#         scale = st.slider('Scale', 0.1, 2.0, 1.0, 0.1)
#         img_edit = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR)
#         img_edit = cv2.convertScaleAbs(img_edit, alpha=gamma, beta=brightness)
#         img_edit = cv2.addWeighted(img_edit, 1.0 + contrast / 100.0, img_edit, 0, 0)
#         img_edit = cv2.GaussianBlur(img_edit, (blur * 2 + 1, blur * 2 + 1), 0)
#         if filter_type == 'Grayscale':
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_GRAY2BGR)
#         elif filter_type == 'Sepia':
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2RGB)
#             img_edit = cv2.transform(img_edit, np.array([[0.393, 0.769, 0.189],
#                                                          [0.349, 0.686, 0.168],
#                                                          [0.272, 0.534, 0.131]]))
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_RGB2BGR)
#         elif filter_type == 'Invert':
#             img_edit = cv2.bitwise_not(img_edit)
#         if flip_horizontal:
#             img_edit = cv2.flip(img_edit, 1)
#         if flip_vertical:
#             img_edit = cv2.flip(img_edit, 0)
#         if rotate != 0:
#             (h, w) = img_edit.shape[:2]
#             center = (w / 2, h / 2)
#             M = cv2.getRotationMatrix2D(center, rotate, scale)
#             img_edit = cv2.warpAffine(img_edit, M, (w, h))
#         img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2RGB)
#         st.image(img_edit, caption='Edited Image', use_column_width=True)


# import os.path as osp
# import glob
# import cv2
# import numpy as np
# import torch
# import RRDBNet_arch as arch
# import os
# import streamlit as st

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
# device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# # device = torch.device('cpu')

# model = arch.RRDBNet(3, 3, 64, 23, gc=32)
# model.load_state_dict(torch.load(model_path), strict=True)
# model.eval()
# model = model.to(device)

# st.title('Image Super-Resolution and Editor')

# test_img_folder = st.file_uploader('Upload test images', type=['png', 'jpg'], accept_multiple_files=True)

# if test_img_folder is not None:
#     st.write('Testing...')
#     idx = 0
#     for path in test_img_folder:
#         idx += 1
#         base = osp.splitext(osp.basename(path.name))[0]
#         st.write(idx, base)
#         # read images
#         img = cv2.imdecode(np.fromstring(path.read(), np.uint8), cv2.IMREAD_COLOR)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # convert to BGR format
#         img = img * 1.0 / 255
#         img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#         img_LR = img.unsqueeze(0)
#         img_LR = img_LR.to(device)

#         with torch.no_grad():
#             output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
#         output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
#         output = (output * 255.0).round()
#         cv2.imwrite('results/{:s}_rlt.png'.format(base), output)

#         st.image(output.astype(np.uint8), caption=base, use_column_width=True)

#         # Add photo editor functionality
#         st.write('Edit the image:')
#         brightness = st.slider('Brightness', -100, 100, 0)
#         contrast = st.slider('Contrast', -100, 100, 0)
#         gamma = st.slider('Gamma', 0.1, 10.0, 1.0, 0.1)
#         blur = st.slider('Blur', 0, 10, 0)
#         filter_type = st.selectbox('Filter', ['None', 'Grayscale', 'Sepia', 'Invert', 'Sketch', 'Cartoon', 'Pencil'])
#         flip_horizontal = st.checkbox('Flip Horizontal')
#         flip_vertical = st.checkbox('Flip Vertical')
#         rotate = st.slider('Rotate', -180, 180, 0)
#         scale = st.slider('Scale', 0.1, 2.0, 1.0, 0.1)
#         img_edit = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR)
#         img_edit = cv2.convertScaleAbs(img_edit, alpha=gamma, beta=brightness)
#         img_edit = cv2.addWeighted(img_edit, 1.0 + contrast / 100.0, img_edit, 0, 0)
#         img_edit = cv2.GaussianBlur(img_edit, (blur * 2 + 1, blur * 2 + 1), 0)
#         if filter_type == 'Grayscale':
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_GRAY2BGR)
#         elif filter_type == 'Sepia':
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2RGB)
#             img_edit = cv2.transform(img_edit, np.array([[0.393, 0.769, 0.189],
#                                                          [0.349, 0.686, 0.168],
#                                                          [0.272, 0.534, 0.131]]))
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_RGB2BGR)
#         elif filter_type == 'Invert':
#             img_edit = cv2.bitwise_not(img_edit)
#         elif filter_type == 'Sketch':
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
#             img_edit = cv2.GaussianBlur(img_edit, (21, 21), 0, 0)
#             img_edit = cv2.divide(img_edit, 255 - img_edit, scale=256)
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_GRAY2BGR)
#         elif filter_type == 'Cartoon':
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
#             img_edit = cv2.medianBlur(img_edit, 7)
#             edges = cv2.Laplacian(img_edit, cv2.CV_8U, ksize=5)
#             ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
#             img_edit = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#         elif filter_type == 'Pencil':
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
#             img_edit = cv2.GaussianBlur(img_edit, (21, 21), 0, 0)
#             img_edit = cv2.divide(img_edit, 255 - img_edit, scale=256)
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_GRAY2BGR)
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
#             img_edit = cv2.adaptiveThreshold(img_edit, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_GRAY2BGR)
#         if flip_horizontal:
#             img_edit = cv2.flip(img_edit, 1)
#         if flip_vertical:
#             img_edit = cv2.flip(img_edit, 0)
#         if rotate != 0:
#             (h, w) = img_edit.shape[:2]
#             center = (w / 2, h / 2)
#             M = cv2.getRotationMatrix2D(center, rotate, scale)
#             img_edit = cv2.warpAffine(img_edit, M, (w, h))
#         img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2RGB)
#         st.image(img_edit, caption='Edited Image', use_column_width=True)


# import os.path as osp
# import glob
# import cv2
# import numpy as np
# import torch
# import RRDBNet_arch as arch
# import os
# import streamlit as st
# from streamlit_cropper import st_cropper

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
# device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# # device = torch.device('cpu')

# model = arch.RRDBNet(3, 3, 64, 23, gc=32)
# model.load_state_dict(torch.load(model_path), strict=True)
# model.eval()
# model = model.to(device)

# st.title('Image Super-Resolution and Editor')

# test_img_folder = st.file_uploader('Upload test images', type=['png', 'jpg'], accept_multiple_files=True)

# def crop_params(image):
#     """Get crop parameters from user input"""
#     x1, y1, x2, y2 = 0, 0, image.shape[1], image.shape[0]
#     if st.button('Select Crop Area'):
#         st.write('Select the area to crop:')
#         cropped_image = st_cropper(image)
#         if cropped_image is not None:
#             y1, x1, y2, x2 = cropped_image
#     return x1, y1, x2, y2

# if test_img_folder is not None:
#     st.write('Testing...')
#     idx = 0
#     for path in test_img_folder:
#         idx += 1
#         base = osp.splitext(osp.basename(path.name))[0]
#         st.write(idx, base)
#         # read images
#         img = cv2.imdecode(np.fromstring(path.read(), np.uint8), cv2.IMREAD_COLOR)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # convert to BGR format
#         img = img * 1.0 / 255
#         img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#         img_LR = img.unsqueeze(0)
#         img_LR = img_LR.to(device)

#         with torch.no_grad():
#             output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
#         output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
#         output = (output * 255.0).round()
#         cv2.imwrite('results/{:s}_rlt.png'.format(base), output)

#         st.image(output.astype(np.uint8), caption=base, use_column_width=True)

#         # Add photo editor functionality
#         st.write('Edit the image:')
#         brightness = st.slider('Brightness', -100, 100, 0)
#         contrast = st.slider('Contrast', -100, 100, 0)
#         gamma = st.slider('Gamma', 0.1, 10.0, 1.0, 0.1)
#         blur = st.slider('Blur', 0, 50, 0)
#         filter_type = st.selectbox('Filter', ['None', 'Grayscale', 'Sepia', 'Invert', 'Sketch', 'Cartoon', 'Pencil'])
#         flip_horizontal = st.checkbox('Flip Horizontal')
#         flip_vertical = st.checkbox('Flip Vertical')
#         rotate = st.slider('Rotate', -180, 180, 0)
#         scale = st.slider('Scale', 0.1, 2.0, 1.0, 0.1)
#         crop = st.checkbox('Crop')
#         if crop:
#             x1, y1, x2, y2 = crop_params(output)
#             img_edit = output[y1:y2, x1:x2]
#         else:
#             img_edit = output
#         img_edit = cv2.cvtColor(img_edit.astype(np.uint8), cv2.COLOR_RGB2BGR)
#         img_edit = cv2.convertScaleAbs(img_edit, alpha=gamma, beta=brightness)
#         img_edit = cv2.addWeighted(img_edit, 1.0 + contrast / 100.0, img_edit, 0, 0)
#         img_edit = cv2.GaussianBlur(img_edit, (blur * 2 + 1, blur * 2 + 1), 0)
#         if filter_type == 'Grayscale':
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_GRAY2BGR)
#         elif filter_type == 'Sepia':
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2RGB)
#             img_edit = cv2.transform(img_edit, np.array([[0.393, 0.769, 0.189],
#                                                          [0.349, 0.686, 0.168],
#                                                          [0.272, 0.534, 0.131]]))
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_RGB2BGR)
#         elif filter_type == 'Invert':
#             img_edit = cv2.bitwise_not(img_edit)
#         elif filter_type == 'Sketch':
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
#             img_edit = cv2.GaussianBlur(img_edit, (21, 21), 0, 0)
#             img_edit = cv2.divide(img_edit, 255 - img_edit, scale=256)
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_GRAY2BGR)
#         elif filter_type == 'Cartoon':
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
#             img_edit = cv2.medianBlur(img_edit, 7)
#             edges = cv2.Laplacian(img_edit, cv2.CV_8U, ksize=5)
#             ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
#             img_edit = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#         elif filter_type == 'Pencil':
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
#             img_edit = cv2.GaussianBlur(img_edit, (21, 21), 0, 0)
#             img_edit = cv2.divide(img_edit, 255 - img_edit, scale=256)
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_GRAY2BGR)
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
#             img_edit = cv2.adaptiveThreshold(img_edit, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
#             img_edit = cv2.cvtColor(img_edit, cv2.COLOR_GRAY2BGR)
#         if flip_horizontal:
#             img_edit = cv2.flip(img_edit, 1)
#         if flip_vertical:
#             img_edit = cv2.flip(img_edit, 0)
#         if rotate != 0:
#             (h, w) = img_edit.shape[:2]
#             center = (w / 2, h / 2)
#             M = cv2.getRotationMatrix2D(center, rotate, scale)
#             img_edit = cv2.warpAffine(img_edit, M, (w, h))
#         img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2RGB)
#         st.image(img_edit, caption='Edited Image', use_column_width=True)


import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import os
import streamlit as st
from streamlit_cropper import st_cropper

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

st.title('Image Super-Resolution and Editor')

test_img_folder = st.file_uploader('Upload test images', type=['png', 'jpg'], accept_multiple_files=True)

def crop_params(image):
    """Get crop parameters from user input"""
    x1, y1, x2, y2 = 0, 0, image.shape[1], image.shape[0]
    if st.button('Select Crop Area'):
        st.write('Select the area to crop:')
        cropped_image = st_cropper(image)
        if cropped_image is not None:
            y1, x1, y2, x2 = cropped_image
    return x1, y1, x2, y2

def merge_images(img1, img2):
    """Merge two images horizontally"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 != h2:
        # resize the images to have the same height
        scale = h1 / h2 if h1 < h2 else h2 / h1
        img1 = cv2.resize(img1, (int(w1 * scale), int(h1 * scale)))
        img2 = cv2.resize(img2, (int(w2 * scale), int(h2 * scale)))
    return np.concatenate((img1, img2), axis=1)

if test_img_folder is not None:
    st.write('Testing...')
    idx = 0
    for path in test_img_folder:
        idx += 1
        base = osp.splitext(osp.basename(path.name))[0]
        st.write(idx, base)
        # read images
        img = cv2.imdecode(np.fromstring(path.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # convert to BGR format
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite('results/{:s}_rlt.png'.format(base), output)

        st.image(output.astype(np.uint8), caption=base, use_column_width=True)

        # Add photo editor functionality
        st.write('Edit the image:')
        brightness = st.slider('Brightness', -100, 100, 0)
        contrast = st.slider('Contrast', -100, 100, 0)
        gamma = st.slider('Gamma', 0.1, 10.0, 1.0, 0.1)
        blur = st.slider('Blur', 0, 100, 0)
        filter_type = st.selectbox('Filter', ['None', 'Grayscale', 'Sepia', 'Invert', 'Sketch', 'Cartoon', 'Pencil'])
        flip_horizontal = st.checkbox('Flip Horizontal')
        flip_vertical = st.checkbox('Flip Vertical')
        rotate = st.slider('Rotate', -180, 180, 0)
        scale = st.slider('Scale', 0.1, 2.0, 1.0, 0.1)
        crop = st.checkbox('Crop')
        if crop:
            x1, y1, x2, y2 = crop_params(output)
            img_edit = output[y1:y2, x1:x2]
        else:
            img_edit = output
        img_edit = cv2.cvtColor(img_edit.astype(np.uint8), cv2.COLOR_RGB2BGR)
        img_edit = cv2.convertScaleAbs(img_edit, alpha=gamma, beta=brightness)
        img_edit = cv2.addWeighted(img_edit, 1.0 + contrast / 100.0, img_edit, 0, 0)
        img_edit = cv2.GaussianBlur(img_edit, (blur * 2 + 1, blur * 2 + 1), 0)
        if filter_type == 'Grayscale':
            img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
            img_edit = cv2.cvtColor(img_edit, cv2.COLOR_GRAY2BGR)
        elif filter_type == 'Sepia':
            img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2RGB)
            img_edit = cv2.transform(img_edit, np.array([[0.393, 0.769, 0.189],
                                                         [0.349, 0.686, 0.168],
                                                         [0.272, 0.534, 0.131]]))
            img_edit = cv2.cvtColor(img_edit, cv2.COLOR_RGB2BGR)
        elif filter_type == 'Invert':
            img_edit = cv2.bitwise_not(img_edit)
        elif filter_type == 'Sketch':
            img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
            img_edit = cv2.GaussianBlur(img_edit, (21, 21), 0, 0)
            img_edit = cv2.divide(img_edit, 255 - img_edit, scale=256)
            img_edit = cv2.cvtColor(img_edit, cv2.COLOR_GRAY2BGR)
        elif filter_type == 'Cartoon':
            img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
            img_edit = cv2.medianBlur(img_edit, 7)
            edges = cv2.Laplacian(img_edit, cv2.CV_8U, ksize=5)
            ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
            img_edit = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        elif filter_type == 'Pencil':
            img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
            img_edit = cv2.GaussianBlur(img_edit, (21, 21), 0, 0)
            img_edit = cv2.divide(img_edit, 255 - img_edit, scale=256)
            img_edit = cv2.cvtColor(img_edit, cv2.COLOR_GRAY2BGR)
            img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
            img_edit = cv2.adaptiveThreshold(img_edit, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
            img_edit = cv2.cvtColor(img_edit, cv2.COLOR_GRAY2BGR)
        if flip_horizontal:
            img_edit = cv2.flip(img_edit, 1)
        if flip_vertical:
            img_edit = cv2.flip(img_edit, 0)
        if rotate != 0:
            (h, w) = img_edit.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, rotate, scale)
            img_edit = cv2.warpAffine(img_edit, M, (w, h))
        img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2RGB)

        # Add merge functionality
        st.write('Merge with another image:')
        merge_img = st.file_uploader('Upload image to merge', type=['png', 'jpg'])
        if merge_img is not None:
            merge_img = cv2.imdecode(np.fromstring(merge_img.read(), np.uint8), cv2.IMREAD_COLOR)
            merge_img = cv2.cvtColor(merge_img, cv2.COLOR_RGB2BGR)
            merge_img = cv2.resize(merge_img, (img_edit.shape[1], img_edit.shape[0]))
            img_edit = merge_images(img_edit, merge_img)
        st.image(img_edit, caption='Edited Image', use_column_width=True)
