########################################################################
#
# Copyright (c) 2017, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
	Read SVO sample to read the video and the information of the camera. It can pick a frame of the svo and save it as
	a JPEG or PNG file. Depth map and Point Cloud can also be saved into files.
"""
import sys
import pyzed.camera as zcam
import pyzed.types as tp
import pyzed.core as core
import pyzed.defines as sl
import cv2
import os
import math






def main():

	# sys.argv[1] = '/home/ogai1234/Desktop/la.svo'
	if len(sys.argv) != 2:
		print("Please specify path to .svo file.")
		exit()

	filepath = sys.argv[1]
	# global n
	n = 1


	print("Reading SVO file: {0}".format(filepath))

	init = zcam.PyInitParameters(svo_input_filename=filepath,svo_real_time_mode=False)
	cam = zcam.PyZEDCamera()
	status = cam.open(init)
	if status != tp.PyERROR_CODE.PySUCCESS:
		print(repr(status))
		exit()

	runtime = zcam.PyRuntimeParameters()
	mat = core.PyMat()
	m = '2018_0751_'
	i = 0
	key = ''
	while (key != 113) and (i < 21000):
		if cam.grab(runtime) == tp.PyERROR_CODE.PySUCCESS:
			cam.retrieve_image(mat)
			cv2.imshow("ZED", mat.get_data())
			if i%10 == 0:
				filepath ='/home/ogai1234/Desktop/p/' + m + str(int(math.floor(i/10))) + '.jpg'
				img = mat.write(filepath)
			key = cv2.waitKey(1)
			# print('11111111111111111111')
			n = n +1
			i = i + 1
	cam.close()
	print("\nFINISH")


def print_camera_information(cam):
	while True:
		res = input("Do you want to display camera information? [y/n]: ")
		if res == "y":
			print()
			print(repr((cam.get_self_calibration_state())))
			print("Distorsion factor of the right cam before calibration: {0}.".format(
				cam.get_camera_information().calibration_parameters_raw.right_cam.disto))
			print("Distorsion factor of the right cam after calibration: {0}.\n".format(
				cam.get_camera_information().calibration_parameters.right_cam.disto))

			print("Confidence threshold: {0}".format(cam.get_confidence_threshold()))
			print("Depth min and max range values: {0}, {1}".format(cam.get_depth_min_range_value(),
																	cam.get_depth_max_range_value()))

			print("Resolution: {0}, {1}.".format(round(cam.get_resolution().width, 2), cam.get_resolution().height))
			print("Camera FPS: {0}".format(cam.get_camera_fps()))
			print("Frame count: {0}.\n".format(cam.get_svo_number_of_frames()))
			break
		elif res == "n":
			print("Camera information not displayed.\n")
			break
		else:
			print("Error, please enter [y/n].\n")


def saving_image(key, mat):


	if key == 115:
		img = tp.PyERROR_CODE.PyERROR_CODE_FAILURE
		while img != tp.PyERROR_CODE.PySUCCESS:
			# filepath = input("Enter filepath name: ")
			# nonlocal n
			filepath ='/home/ogai1234/Desktop/piccc/' + m + str(n) + '.jpg'
			img = mat.write(filepath)
			print('11111111111111111111')
			# print("Saving image : {0}".format(repr(img)))
			n = n + 1
			if img == tp.PyERROR_CODE.PySUCCESS:
				break
			else:
				print("Help: you must enter the filepath + filename + PNG extension.")


def saving_depth(cam):
	while True:
		res = input("Do you want to save the depth map? [y/n]: ")
		if res == "y":
			save_depth = 0
			while not save_depth:
				filepath = input("Enter filepath name: ")
				save_depth = zcam.save_camera_depth_as(cam, sl.PyDEPTH_FORMAT.PyDEPTH_FORMAT_PNG, filepath)
				if save_depth:
					print("Depth saved.")
					break
				else:
					print("Help: you must enter the filepath + filename without extension.")
			break
		elif res == "n":
			print("Depth will not be saved.")
			break
		else:
			print("Error, please enter [y/n].")


def saving_point_cloud(cam):
	while True:
		res = input("Do you want to save the point cloud? [y/n]: ")
		if res == "y":
			save_point_cloud = 0
			while not save_point_cloud:
				filepath = input("Enter filepath name: ")
				save_point_cloud = zcam.save_camera_point_cloud_as(cam,
																   sl.PyPOINT_CLOUD_FORMAT.
																   PyPOINT_CLOUD_FORMAT_PCD_ASCII,
																   filepath, True)
				if save_point_cloud:
					print("Point cloud saved.")
					break
				else:
					print("Help: you must enter the filepath + filename without extension.")
			break
		elif res == "n":
			print("Point cloud will not be saved.")
			break
		else:
			print("Error, please enter [y/n].")

if __name__ == "__main__":
	main()
