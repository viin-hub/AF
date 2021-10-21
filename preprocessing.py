import os
from glob import glob
import os
from pathlib import Path
import pandas as pd
import shutil
import nibabel as nib
import matplotlib.pyplot as plt
import pydicom
from pydicom import dcmread
from scipy import ndimage
import numpy as np
from nibabel import nifti1 as nitool


def inspect_file():

	af_folders = glob("/home/miranda/Documents/PhD/Angelos/Shared/segmentedimages/AF/*/")
	af_mis = []
	for f in af_folders:
		contents = os.listdir(f)
		if 'DICOM' in contents:
			pass
		else:
			parts = f.split('/')
			af_mis.append(parts[-2])


	print(af_mis)

	control_folders = glob("/home/miranda/Documents/PhD/Angelos/Shared/segmentedimages/control/*/")
	control_mis = []
	for f in control_folders:
		contents = os.listdir(f)
		if 'DICOM' in contents:
			pass
		else:
			parts = f.split('/')
			control_mis.append(parts[-2])
	print(control_mis)


def copy_rename():

	trg_1 = '/home/miranda/Documents/data/AF/AF/'
	trg_2 = '/home/miranda/Documents/data/AF/Control/'

	df = pd.read_csv('/home/miranda/Documents/PhD/Angelos/Shared/segmentedimages/nii_files.csv',delimiter='\t')
	for index, row in df.iterrows():
		old_name = row[0]
		parts = old_name.split('/')
		fn = parts[-3]
		group = parts[-4]
		if group == 'control':
			new_name = os.path.join(trg_2,fn+'.nii')
			if not os.path.exists(new_name):
				shutil.copy(old_name, new_name)
		elif group == 'AF':
			new_name = os.path.join(trg_1,fn+'.nii')
			if not os.path.exists(new_name):
				shutil.copy(old_name, new_name)

def nii_file_format():

	df = pd.read_csv('/home/miranda/Documents/PhD/Angelos/Shared/segmentedimages/nii_files.csv',delimiter='\t')
	trg_1 = '/home/miranda/Documents/data/AF/AF/'
	trg_2 = '/home/miranda/Documents/data/AF/Control/'

	na = []
	sh = []
	new_name2 = []
	for index, row in df.iterrows():
		old_name = row[0]
		parts = old_name.split('/')
		fn = parts[-3]
		group = parts[-4]
		ni = nib.load(old_name)
		data = ni.get_fdata()
		na.append(old_name)
		sh.append([data.shape])
		image = data[:,:,:,1]
		new_image = nib.Nifti1Image(image, affine=ni.affine)
		# nib.save(new_image, os.path.join(folder,basename+'_'+'norm'+'.nii.gz'))
		new_name = os.path.join(trg_1,fn+'.nii')
		if group == 'control':
			new_name = os.path.join(trg_2,fn+'.nii.gz')
			if not os.path.exists(new_name):
				nib.save(new_image, new_name)
				new_name2.append(new_name)
		elif group == 'AF':
			new_name = os.path.join(trg_1,fn+'.nii.gz')
			if not os.path.exists(new_name):
				nib.save(new_image, new_name)
				new_name2.append(new_name)
	d = {'NAME': na,'SHAPE':sh} 
	df = pd.DataFrame(data=d)
	df.to_csv('/home/miranda/Documents/PhD/Angelos/Shared/segmentedimages/nii_shape.csv',index=False)

	df_img = pd.DataFrame(new_name2)
	df_img.to_csv('/home/miranda/Documents/data/AF/nii.csv',index=False)


def convert_mgz2nii():

	df = pd.read_csv('/home/miranda/Documents/data/AF/AF_v2.csv')
	for index, row in df.iterrows():
		f = row[0]
		nid = os.path.basename(os.path.splitext(f)[0])
		ni = nib.load(f)
		data = ni.get_fdata()
		new_image = nib.Nifti1Image(data, affine=ni.affine)
		if 'brain' in nid:
			new_name = os.path.join('/home/miranda/Documents/data/AF/V3/AF_v3/brain',nid + '.nii.gz')
			nib.save(new_image, new_name)
		elif 'wm' in nid:
			new_name = os.path.join('/home/miranda/Documents/data/AF/V3/AF_v3/wm',nid + '.nii.gz')
			nib.save(new_image, new_name)

	df1 = pd.read_csv('/home/miranda/Documents/data/AF/Control_v2.csv')
	for index, row in df1.iterrows():
		f = row[0]
		nid = os.path.basename(os.path.splitext(f)[0])
		ni = nib.load(f)
		data = ni.get_fdata()
		new_image = nib.Nifti1Image(data, affine=ni.affine)
		if 'brain' in nid:
			new_name = os.path.join('/home/miranda/Documents/data/AF/V3/Control_v3/brain',nid + '.nii.gz')
			nib.save(new_image, new_name)
		elif 'wm' in nid:
			new_name = os.path.join('/home/miranda/Documents/data/AF/V3/Control_v3/wm',nid + '.nii.gz')
			nib.save(new_image, new_name)

def split_csv():
	df = pd.read_csv('/home/miranda/Documents/data/AF/nii.csv')
	# print(df.shape) # 81, 1

	df_1 = df.loc[0:10]
	df_2 = df.loc[11:21]
	df_3 = df.loc[22:33]
	df_4 = df.loc[34:45]
	df_5 = df.loc[46:57]
	df_6 = df.loc[58:69]
	df_7 = df.loc[70:81]
	
	df_1.to_csv('/home/miranda/Documents/data/AF/nii_1.csv',index=False)
	df_2.to_csv('/home/miranda/Documents/data/AF/nii_2.csv',index=False)
	df_3.to_csv('/home/miranda/Documents/data/AF/nii_3.csv',index=False)
	df_4.to_csv('/home/miranda/Documents/data/AF/nii_4.csv',index=False)
	df_5.to_csv('/home/miranda/Documents/data/AF/nii_5.csv',index=False)
	df_6.to_csv('/home/miranda/Documents/data/AF/nii_6.csv',index=False)
	df_7.to_csv('/home/miranda/Documents/data/AF/nii_7.csv',index=False)

def fs_data():

	# get freesufer processed brain and wm images

	#  reference: AF and control subject names
	df_af = pd.read_csv('/home/miranda/Documents/data/AF/AF_v1.csv')
	df_c = pd.read_csv('/home/miranda/Documents/data/AF/Control_v1.csv')
	fn_af = []
	for index, row in df_af.iterrows():
		f = row[0]
		bs = os.path.basename(os.path.splitext(os.path.splitext(f)[0])[0])
		fn_af.append(bs)

	fn_c = []
	for index, row in df_c.iterrows():
		f = row[0]
		bs = os.path.basename(os.path.splitext(os.path.splitext(f)[0])[0])
		fn_c.append(bs)

	# target folders
	t_af = '/home/miranda/Documents/data/AF/freesurfer/AF_v2'
	t_c = '/home/miranda/Documents/data/AF/freesurfer/Control_v2'

	folders = glob("/usr/local/freesurfer/subjects/*")

	for f in folders:
		a = os.path.basename(f)
		if a in fn_af:
			if os.path.isfile(os.path.join(f, "mri", "brain.mgz")) and os.path.isfile(os.path.join(f, "mri", "wm.mgz")):
				shutil.copy(os.path.join(f, "mri", "brain.mgz"), os.path.join(t_af, a + "_brain.mgz"))
				shutil.copy(os.path.join(f, "mri", "wm.mgz"), os.path.join(t_af, a + "_wm.mgz"))
		elif a in fn_c:
			if os.path.isfile(os.path.join(f, "mri", "brain.mgz")) and os.path.isfile(os.path.join(f, "mri", "wm.mgz")):
				shutil.copy(os.path.join(f, "mri", "brain.mgz"), os.path.join(t_c, a + "_brain.mgz"))
				shutil.copy(os.path.join(f, "mri", "wm.mgz"), os.path.join(t_c, a + "_wm.mgz"))
			

def qc_data():

	af_f = glob('/home/miranda/Documents/data/AF/V3/Control_v3/brain/*')

	for f in af_f:
		scan = nib.load(f)
		scan = scan.get_fdata()
		plt.imshow(scan[100, :, :], cmap="gray")
		elements = f.split('/')
		name = elements[-1]
		basename = name.replace('.nii.gz', '') 
		plt.savefig('/home/miranda/Documents/data/AF/V3/Control_v3/pic/'+basename+'.png')



def img_meta():

	list_mod = []
	list_date = []
	list_id = []
	list_pid = []
	list_int = []
	list_sex = []
	list_age = []
	list_at = []
	list_sd = []
	list_ssd = []
	list_fpath = []
	list_protocol = []
	list_manuf = []
	list_pix = []
	list_date = []
	list_time = []
	list_label = []

	af_folders = glob("/home/miranda/Documents/PhD/Angelos/Shared/segmentedimages/AF/*/")
	af_mis = []
	for f in af_folders:
		contents = os.listdir(f)
		if 'DICOM' in contents:
			current_f = os.path.join(f, 'DICOM', 'Image-00001')
			# dcm = glob(current_folder+'/*',recursive=True)
			if os.path.isfile(current_f):
				ds = pydicom.dcmread(current_f)
				try:
					mod = ds.Modality
				except:
					mod = []
				try:
					nam = ds.PatientName 
				except:
					nam = []
				try:						
					pid = ds.PatientID 
				except:	
					pid = []
				try:
					pin = ds.InstitutionName 
				except:
					pin = []
				try:
					sex = ds.PatientSex
				except:
					sex = []
				try:
					age = ds.PatientAge
					age = age.replace("0", "") 
					age = age.replace("Y", "") 
					age = int(age)
				except:
					age = []
				try:
					sd = ds.StudyDescription
				except:
					sd = []
				try:
					ssd = ds.SeriesDescription
				except:
					ssd = []
				try:
					protocol = ds.ProtocolName
				except:
					protocol = []
				try:
					manuf = ds.Manufacturer
				except:	
					manuf = []	
				try:
					acq_date = ds.AcquisitionDate
				except:
					acq_date = []
				try:
					at = ds.AcquisitionTime
				except:
					at = []
				try:
					pix = ds.PixelSpacing
				except:
					pix = []
				
			
				list_mod.append(mod)
				list_id.append(nam)
				list_pid.append(pid)
				list_int.append(pin)
				list_sex.append(sex)
				list_age.append(age)
				list_sd.append(sd)
				list_ssd.append(ssd)
				list_fpath.append(current_f)
				list_protocol.append(protocol)
				list_manuf.append(manuf)
				list_date.append(acq_date)
				list_time.append(at)
				list_pix.append(pix)
				list_label.append('AF')




	control_folders = glob("/home/miranda/Documents/PhD/Angelos/Shared/segmentedimages/control/*/")
	control_mis = []
	for f in control_folders:
		contents = os.listdir(f)
		if 'DICOM' in contents:
			current_f = os.path.join(f, 'DICOM', 'Image-00001')
			if os.path.isfile(current_f):
				ds = pydicom.dcmread(current_f)
				try:
					mod = ds.Modality
				except:
					mod = []
				try:
					nam = ds.PatientName 
				except:
					nam = []
				try:						
					pid = ds.PatientID 
				except:	
					pid = []
				try:
					pin = ds.InstitutionName 
				except:
					pin = []
				try:
					sex = ds.PatientSex
				except:
					sex = []
				try:
					age = ds.PatientAge
					age = age.replace("0", "") 
					age = age.replace("Y", "") 
					age = int(age)
				except:
					age = []
				try:
					sd = ds.StudyDescription
				except:
					sd = []
				try:
					ssd = ds.SeriesDescription
				except:
					ssd = []
				try:
					protocol = ds.ProtocolName
				except:
					protocol = []
				try:
					manuf = ds.Manufacturer
				except:	
					manuf = []	
				try:
					acq_date = ds.AcquisitionDate
				except:
					acq_date = []
				try:
					at = ds.AcquisitionTime
				except:
					at = []
				try:
					pix = ds.PixelSpacing
				except:
					pix = []
				
			
				list_mod.append(mod)
				list_id.append(nam)
				list_pid.append(pid)
				list_int.append(pin)
				list_sex.append(sex)
				list_age.append(age)
				list_sd.append(sd)
				list_ssd.append(ssd)
				list_fpath.append(current_f)
				list_protocol.append(protocol)
				list_manuf.append(manuf)
				list_date.append(acq_date)
				list_time.append(at)
				list_pix.append(pix)
				list_label.append('Control')
	

	d = {'ID': list_id,'PTID':list_pid, 'Age':list_age, 'Sex':list_sex,'Image_Modality': list_mod, \
	'Manufacturer': list_manuf, 'Protocol': list_protocol, 'Institution': list_int, \
	'Study_Description':list_sd, 'Series_Description':list_ssd, 'AcquisitionDate':list_date,\
	'AcquisitionTime':list_time, 'Pixel Spacing': list_pix,'Label': list_label,'Path':list_fpath} 
	df_img = pd.DataFrame(data=d)

	df_img.to_csv('/home/miranda/Documents/data/AF/meta.csv',index=False)

	df_af = df_img.loc[df_img['Label']=='AF']
	df_con = df_img.loc[df_img['Label']=='Control']

	print('AF')
	print('==================')
	print('Number:', len(df_af.index))
	print('Age:', df_af['Age'].mean())
	print('Age std:', df_af['Age'].std())
	print('Sex:', df_af['Sex'].value_counts())

	print('Control')
	print('==================')
	print('Number:', len(df_con.index))
	print('Age:', df_con['Age'].mean())
	print('Age std:', df_con['Age'].std())
	print('Sex:', df_con['Sex'].value_counts())

def spm_nii():

	df1 = pd.read_csv('/home/miranda/Documents/data/AF/AF_v3_qc.csv')
	for index, row in df1.iterrows():
		f = row[0]
		n = os.path.basename(os.path.splitext(f)[0])
		ni = nib.load(f)
		data = ni.get_fdata()
		new_image = nib.Nifti1Image(data, affine=ni.affine)
		new_name = os.path.join('/home/miranda/Documents/data/AF/V8/AF_v8/brain',n)
		nib.save(new_image, new_name)

def resize_volume(img, desired_depth, desired_width, desired_height):
	"""Resize across z-axis"""
	# Get current depth
	current_depth = img.shape[-1]
	current_width = img.shape[0]
	current_height = img.shape[1]
	# Compute depth factor
	depth = current_depth / desired_depth
	width = current_width / desired_width
	height = current_height / desired_height
	depth_factor = 1 / depth
	width_factor = 1 / width
	height_factor = 1 / height
	# Rotate
	img = ndimage.rotate(img, 90, reshape=False)
	# Resize across z-axis
	img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
	return img

def crop_image():
	# crop and resize to 128, 128, 128

	df_1 = pd.read_csv('/home/miranda/Documents/data/AF/AF_v203.csv', header=None)
	df_2 = pd.read_csv('/home/miranda/Documents/data/AF/Control_v203.csv', header=None)

	# af_scan_paths = [
	# 	    os.path.join("/home/miranda/Documents/data/AF/V3/AF_v3/brain", x)
	# 	    for x in os.listdir("/home/miranda/Documents/data/AF/V3/AF_v3/brain")
	# 	]

	# con_scan_paths = [
	# 	    os.path.join("/home/miranda/Documents/data/AF/V3/Control_v3/brain", x)
	# 	    for x in os.listdir("/home/miranda/Documents/data/AF/V3/Control_v3/brain")
	# 	]

	t1 = "/home/miranda/Documents/data/AF/V2.4/AF"
	t2 = "/home/miranda/Documents/data/AF/V2.4/Control"

	for index, row in df_1.iterrows():
	# for f in af_scan_paths:
		f = row[0]
		scan = nib.load(f)
		affine = scan.affine
		img = scan.get_fdata()
		nid = os.path.basename(f)
		if "*" not in nid:
			# Create a mask with the background pixels
			mask = img == 0

			# Find the brain area
			coords = np.array(np.nonzero(~mask))
			top_left = np.min(coords, axis=1) 
			bottom_right = np.max(coords, axis=1) 

			# Remove the background
			croped_image = img[top_left[0]+3:bottom_right[0]-3, top_left[1]+3:bottom_right[1]-3, top_left[2]+3:bottom_right[2]-3]
			img1 = resize_volume(img, 128, 128, 128)
			new_image = nib.Nifti1Image(img1, affine=affine)
			nib.save(new_image, os.path.join(t1,nid))

	for index, row in df_2.iterrows():
	# for f in con_scan_paths:
		f = row[0]
		scan = nib.load(f)
		affine = scan.affine
		img = scan.get_fdata()
		nid = os.path.basename(f)
		if "*" not in nid:
			# Create a mask with the background pixels
			mask = img == 0

			# Find the brain area
			coords = np.array(np.nonzero(~mask))
			top_left = np.min(coords, axis=1) 
			bottom_right = np.max(coords, axis=1) 

			# Remove the background
			croped_image = img[top_left[0]+3:bottom_right[0]-3, top_left[1]+3:bottom_right[1]-3, top_left[2]+3:bottom_right[2]-3]
			img1 = resize_volume(img, 128, 128, 128)
			new_image = nib.Nifti1Image(img1, affine=affine)
			nib.save(new_image, os.path.join(t2,nid))

def mv_file():

	af_d = '/home/miranda/Documents/data/AF/V2.2/AF'

	af_files = glob('/home/miranda/Documents/data/AF/AF/*')
	for f in af_files:
		print(f)
		if '_brain' in f:
			shutil.move(f, af_d)
		if '_cropped' in f:
			shutil.move(f, af_d)
		if '_or' in f:
			shutil.move(f, af_d)
			
def mv_file2():

	df1 = pd.read_csv('/home/miranda/Documents/data/AF/v21_bet.txt', header=None, sep='\t')
	save_dir = '/home/miranda/Documents/data/AF/V3.1'

	for index, row in df1.iterrows():
		f = row[0]
		name = os.path.basename(f)
		parts = f.split("/")
		# print(parts)
		fn = parts[-4]
		fn1 = parts[-3]
		fn2 = parts[-2]
		targ_dir = os.path.join(save_dir,fn, fn1, fn2)
		print(targ_dir)
		if not os.path.exists(targ_dir):
			os.makedirs(targ_dir)

		shutil.move(f, os.path.join(targ_dir,name))


def rm_f():
	df = pd.read_csv('/home/miranda/Documents/data/AF/V1_nii.txt', header=None, sep='\n')

	trg_dir = '/home/miranda/Documents/data/AF/V2.6'
	list_f = []
	list_dim = []
	for index, row in df.iterrows():
		f = row[0]
		if 'Localiser' not in f:
			list_f.append(f)
			img = nib.load(f)
			data = img.get_fdata()
			if len(data.shape) == 4:
				data_1 = data[:,:,:,0]
				img_1 = nitool.Nifti1Pair(data_1,img.affine)
				parts = f.split("/")
				# print(parts)
				fn = parts[-3]
				fn1 = parts[-2]
				targ_dir = os.path.join(trg_dir,fn,fn1)
				if not os.path.exists(targ_dir):
					os.makedirs(targ_dir)
				save_name = os.path.join(targ_dir, 'dwi'+'.nii.gz')
				nitool.save(img_1, save_name)
			elif len(data.shape) == 3:
				shutil.copy(f, os.path.join(targ_dir, 'dwi'+'.nii.gz'))
			list_dim.append(data.shape)

	d = {'file':list_f, 'dim':list_dim}
	df_new = pd.DataFrame(data = d)
	df_new.to_csv('/home/miranda/Documents/data/AF/V1_nii2.csv', index = False)

def dcm_series():

	df = pd.read_csv('/home/miranda/Documents/PhD/Angelos/angelos_sept_2021/files.txt', header=None, sep='\n')

	# DWI Ax Resolve_TRACEW

	list_file = []
	list_parent_folder = []
	list_series = []	
	list_pn = []
	list_pid = []
	list_dob = []
	list_sex = []
	list_age = []
	list_fs = []
	list_space = []
	list_ps = []
	for index, row in df.iterrows():
		f = row[0]
		list_file.append(f)
		folder = os.path.dirname(f)
		list_parent_folder.append(folder)
		ds = dcmread(f)
		try:
			sd = ds.SeriesDescription
			list_series.append(sd)
		except:
			list_series.append(np.nan)
		try:
			pn = ds.PatientName
			list_pn.append(pn)
		except:
			list_pn.append(np.nan)
		try:
			pid = ds.PatientID
			list_pid.append(pid)
		except:
			list_pid.append(np.nan)
		try:
			dob = ds.PatientBirthDate
			list_dob.append(dob)
		except:
			list_dob.append(np.nan)
		try:
			sex = ds.PatientSex
			list_sex.append(sex)
		except:
			list_sex.append(np.nan)
		try:
			age = ds.PatientAge
			list_age.append(age)
		except:
			list_age.append(np.nan)
		try:
			fs = ds.MagneticFieldStrength
			list_fs.append(fs)
		except:
			list_fs.append(np.nan)
		try:
			space = ds.SpacingBetweenSlices
			list_space.append(space)
		except:
			list_space.append(np.nan)
		try:
			ps = ds.PixelSpacing
			list_ps.append(ps)
		except:
			list_ps.append(np.nan)
			

	d = {'PatientName':list_pn, 'PatientID':list_pid, 'BirthDate':list_dob,\
	'Sex':list_sex, 'Age':list_age, 'Series':list_series, 'FieldStrength':list_fs,\
	'SliceThickness':list_space, 'PixelSpacing':list_ps, 'File':list_file}

	df_nw = pd.DataFrame(data=d)
	df_nw.to_csv('/home/miranda/Documents/PhD/Angelos/angelos_sept_2021/files_info.csv',index=False)

def dwi_b1000():

	df = pd.read_csv('/home/miranda/Documents/PhD/Angelos/angelos_PhD_sept_27_images/nii.txt', header=None, sep='\n')


	for index, row in df.iterrows():
		f = row[0]
		f0 = f.replace('DICOM/', '')
		f_new = f0.replace('/home/miranda/Documents/PhD/Angelos/angelos_PhD_sept_27_images', '/home/miranda/Documents/data/AF/V1')
		d = os.path.dirname(f_new)
		d_new = d.replace('V1', 'V2.1')
		if not os.path.exists(d_new):
			os.makedirs(d_new)
		img = nib.load(f_new)
		data = img.get_fdata()
		print(f_new)
		print(data.shape)
		if f_new == "/home/miranda/Documents/data/AF/V1/AF/patient8_2018837_AF_brain/diffusion-weighted_axial_6.nii":
			v_b1000 = data
		else:
			v_b1000 = data[:,:,:,1]
		new_image = nib.Nifti1Image(v_b1000, affine=img.affine)
		new_name = os.path.join(d_new,'DWI_b1000.nii')
		nib.save(new_image,new_name)



# inspect_file()
# copy_rename()
# nii_file_format()
# split_csv()
# fs_data()
# convert_mgz2nii()
# qc_data()
# img_meta()
# spm_nii()
# crop_image()
# mv_file()
# mv_file2()
# rm_f()
# dcm_series()
# dwi_b1000()