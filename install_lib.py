import subprocess
import sys
import re
import ctypes
import platform
import argparse
import os

def cuda_version_nvcc():
	try:
		output = subprocess.check_output(["nvcc", "--version"], encoding="utf-8")
		match = re.search(r"release (\d+\.\d+)", output)
		print(f"nvcc output: {output}".rstrip())
		if match:
			cuda_version = match.group(1).replace(".", "")
			print(f"NVCC cuda version: {cuda_version}")
			return cuda_version
		else:
			print("Could not find CUDA version in nvcc output.")
			return None
	except Exception as e:
		print(f"Error detecting CUDA version with nvcc: {e}")
		return None

def cuda_version_libcuda():
	try:
		if platform.system() == "Windows":
			libcudart = ctypes.windll.LoadLibrary("nvcuda.dll")
		else:
			libcudart = ctypes.cdll.LoadLibrary("libcuda.so")

		version = ctypes.c_int()
		result = libcudart.cuDriverGetVersion(ctypes.byref(version))

		if result == 0:  # CUDA_SUCCESS
			version_str = str(version.value)
			major = version_str[0:2]
			minor = version_str[2:3]
   
			#major = version.value // 1000
			#minor = (version.value % 1000) // 100
			print(f"cuDriverGetVersion version value {version.value}, extracted cuda version: {major}.{minor}")
			return f"{major}{minor}"
		else:
			print(f"cuDriverGetVersion failed with code {result}")
			return None
	except Exception as e:
		print(f"Error detecting CUDA version with libcuda: {e}")
		return None

def detect_cuda_version(force_cuda=None):
	if force_cuda:
		return force_cuda
	return cuda_version_nvcc() or cuda_version_libcuda()

def pip_install(packages, index_url=None, find_links=None):
	cmd = [sys.executable, "-m", "pip", "install"] + packages
	if index_url:
		cmd += ["--extra-index-url", index_url]
	if find_links:
		cmd += ["-f", find_links]
	subprocess.check_call(cmd)

def install_all(cuda_version):
	if cuda_version == "121":
		print("Installing PyTorch 2.1.0 with CUDA 12.1...")
		pip_install(
			["torch==2.1.0"],
			index_url="https://download.pytorch.org/whl/cu121"
		)
  		# pip install "pytorch_lightning==2.1.2"
		pip_install(["pytorch_lightning==2.1.2"])
  		# pip install torch_geometric
		pip_install(
			["torch_geometric==2.4.0"],
		)
  		# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
		pip_install(
			["torch_scatter==2.1.2"],
			find_links="https://data.pyg.org/whl/torch-2.1.0+cu121.html"
		)
  		# pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
		pip_install(
			["dgl==1.1.3+cu121"],
			find_links="https://data.dgl.ai/wheels/cu121/repo.html"			
		)
	elif cuda_version == "118":
		print("Installing PyTorch 2.1.0 with CUDA 11.8...")
		pip_install(
			["torch==2.1.0"],
			index_url="https://download.pytorch.org/whl/cu118"
		)
		# pip install "pytorch_lightning==2.1.2"
		pip_install(["pytorch_lightning==2.1.2"])
		# pip install torch_geometric
		pip_install(
			["torch_geometric==2.4.0"],
		)
		# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
		pip_install(
			["torch_scatter==2.1.2"],
			find_links="https://data.pyg.org/whl/torch-2.1.0+cu118.html"
		)
		# pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
		pip_install(
			["dgl==1.0.4+cu118"],
			find_links="https://data.dgl.ai/wheels/cu118/repo.html"		
		)
	else:
		print("Installing CPU version of PyTorch 2.1.2...")
		pip_install(
			["torch==2.1.0"],
			index_url="https://download.pytorch.org/whl/cpu"
		)
		pip_install(["pytorch_lightning==2.1.2", "torch_geometric==2.4.0", "dgl==1.0.4"])
		# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cpu.html
		pip_install(
			["torch_scatter==2.1.2"],
			index_url="https://data.pyg.org/whl/torch-2.1.2+cpu.html"
		)		

def main():
	parser = argparse.ArgumentParser(description="Install PyTorch and related libraries.")
	parser.add_argument(
		"--force-cuda",
		type=str,
		default=None,
		choices=["118", "121"],
		help="Force a specific CUDA version (11.8 or 12.1)."
	)
	parser.add_argument(
		"--force-cpu",
		action="store_true",
		help="Force CPU version."
	)
	
	args = parser.parse_args()
 
	assert not (args.force_cuda and args.force_cpu), "Cannot force both CPU and CUDA versions."
	print(f"Force CPU: {args.force_cpu}, Force CUDA: {args.force_cuda}")
	print(f"-" * 30 )
	if args.force_cpu:
		cuda_version = None
	elif args.force_cuda:
		cuda_version = args.force_cuda
	else:
		cuda_version = detect_cuda_version()
	print(f"Detected CUDA version: {cuda_version}" if cuda_version else "CUDA not detected.")
	if not cuda_version:
		print(f"If you are sure cuda 11.8 or 12.1 use --force_cuda <version>")
	print(f"-" * 30 )
	input("Press Enter to continue...")
	install_all(cuda_version)
 
if __name__ == "__main__":
	main() 