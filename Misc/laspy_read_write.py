def read_las(pointcloudfile,get_attributes=False,useevery=1):
	'''

	:param pointcloudfile: specification of input file (format: las or laz)
	:param get_attributes: if True, will return all attributes in file, otherwise will only return XYZ (default is False)
	:param useevery: value specifies every n-th point to use from input, i.e. simple subsampling (default is 1, i.e. returning every point)
	:return: 3D array of points (x,y,z) of length number of points in input file (or subsampled by 'useevery')
	'''

	import laspy
	import numpy as np

	# read the file
	inFile = laspy.read(pointcloudfile)

	# get the coordinates (XYZ)
	coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
	coords = coords[::useevery, :]

	if get_attributes == False:
		return (coords)

	else:
		las_fields= [info.name for info in inFile.points.point_format.dimensions]
		attributes = {}

		for las_field in las_fields[3:]: # skip the X,Y,Z fields
			attributes[las_field] = inFile.points[las_field][::useevery]

		return (coords, attributes)

def write_las(outpoints,outfilepath,attribute_dict={}):

	'''

	:param outpoints: 3D array of points to be written to output file
	:param outfilepath: specification of output file (format: las or laz)
	:param attribute_dict: dictionary of attributes (key: name of attribute; value: 1D array of attribute values in order of points in 'outpoints'); if not specified, dictionary is empty and nothing is added
	:return: None
	'''

	import laspy

	hdr = laspy.LasHeader(version="1.4", point_format=6)
	hdr.x_scale = 0.00025
	hdr.y_scale = 0.00025
	hdr.z_scale = 0.00025
	mean_extent = np.mean(outpoints, axis=0)
	hdr.x_offset = int(mean_extent[0])
	hdr.y_offset = int(mean_extent[1])
	hdr.z_offset = int(mean_extent[2])

	las = laspy.LasData(hdr)

	las.x = outpoints[:, 0]
	las.y = outpoints[:, 1]
	las.z = outpoints[:, 2]
	for key,vals in attribute_dict.items():
		try:
			las[key] = vals
		except:
			las.add_extra_dim(laspy.ExtraBytesParams(
				name=key,
				type=type(vals[0])
				))
			las[key] = vals

	las.write(outfilepath)

	return
