import io
from pathlib import Path

import numpy as np
from construct import Float64n, Float32n, Int8un, Int16un, Int32un, PaddedString, Struct, ListContainer
from PIL import Image

from oct_converter.image_types import FundusImageWithMetaData, OCTVolumeWithMetaData


class FDA(object):
    """Class for extracting data from Topcon's .fda file format.

    Notes:
        Mostly based on description of .fda file format here:
        https://bitbucket.org/uocte/uocte/wiki/Topcon%20File%20Format

    Attributes:
        filepath (str): Path to .img file for reading.
        header (obj:Struct): Defines structure of volume's header.
        oct_header (obj:Struct): Defines structure of OCT header.
        fundus_header (obj:Struct): Defines structure of fundus header.
        chunk_dict (dict): Name of data chunks present in the file, and their start locations.
        hw_info_03_header (obj:Struct) : Defines structure of hw info header
        patient_info_02_header (obj:Struct) : Defines patient info header
        fda_file_info_header (obj:Struct) : Defines fda file info header
        capture_info_02_header (obj:Struct) : Defines capture info header
        param_scan_04_header (obj:Struct) : Defines param scan header
        img_trc_02_header (obj:Struct) : Defines img trc header
        param_obs_02_header (obj:Struct) : Defines param obs header
        img_mot_comp_03_header (obj:Struct) : Defines img mot comp header
        effective_scan_range_header (obj:Struct) : Defines effective scan range header
        regist_info_header (obj:Struct) : Defines regist info header
        result_cornea_curve_header (obj:Struct) : Defines result cornea curve header
        result_cornea_thickness_header (obj:Struct) : Defines result cornea thickness header
        contour_info_header (obj:Struct) : Defines contour info header
        align_info_header (obj:Struct) : Defines align info header
        fast_q2_info_header (obj:Struct) : Defines fast q2 info header
        gla_littmann_01_header (obj : Struct) : Defines gla littmann 01 header
    """

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(self.filepath)
        self.header = Struct(
            "FOCT" / PaddedString(4, "ascii"),
            "FDA" / PaddedString(3, "ascii"),
            "version_info_1" / Int32un,
            "version_info_2" / Int32un,
        )
        self.oct_header = Struct(
            "type" / PaddedString(1, "ascii"),
            "unknown1" / Int32un,
            "unknown2" / Int32un,
            "width" / Int32un,
            "height" / Int32un,
            "number_slices" / Int32un,
            "unknown3" / Int32un,
        )

        self.oct_header_2 = Struct(
            "unknown" / PaddedString(1, "ascii"),
            "width" / Int32un,
            "height" / Int32un,
            "bits_per_pixel" / Int32un,
            "number_slices" / Int32un,
            "unknown" / PaddedString(1, "ascii"),
            "size" / Int32un,
        )

        self.fundus_header = Struct(
            "width" / Int32un,
            "height" / Int32un,
            "bits_per_pixel" / Int32un,
            "number_slices" / Int32un,
            "unknown" / PaddedString(4, "ascii"),
            "size" / Int32un,
            # 'img' / Int8un,
        )

        self.hw_info_03_header = Struct(
            "model_name" / PaddedString(16, "ascii"),
            "serial_number" / PaddedString(16, "ascii"),
            "zeros" / PaddedString(32, "ascii"),
            "version" / PaddedString(16, "ascii"),
            "build_year" / Int16un,
            "build_month" / Int16un,
            "build_day" / Int16un,
            "build_hour" / Int16un,
            "build_minute" / Int16un,
            "build_second" / Int16un,
            "zeros" / PaddedString(8, "ascii"),
            "version_numbers" / PaddedString(8, "ascii"),
        )

        self.patient_info_02_header = Struct(
            "patient_id" / PaddedString(8, "ascii"),
            "patient_given_name" / PaddedString(8, "ascii"),
            "patient_surname" / PaddedString(8, "ascii"),
            "birth_date_type" / Int8un,
            "birth_year" / Int16un,
            "birth_month" / Int16un,
            "birth_day" / Int16un,
        )

        self.fda_file_info_header = Struct(
            "0x2" / Int32un,
            "0x3e8" / Int32un,
            "8.0.1.20198" / PaddedString(32, "ascii"),
        )

        self.capture_info_02_header = Struct(
            "x" / Int16un,
            "zeros" / PaddedString(52, "ascii"),
            "aquisition_year" / Int16un,
            "aquisition_month" / Int16un,
            "aquisition_day" / Int16un,
            "aquisition_hour" / Int16un,
            "aquisition_minute" / Int16un,
            "aquisition_second" / Int16un,
        )

        self.param_scan_04_header = Struct(
            "tomogram_x_dimension_in_mm" / Float64n,
            "tomogram_y_dimension_in_mm" / Float64n,
            "tomogram_z_dimension_in_um" / Float64n,
        )

        self.img_trc_02_header = Struct(
            "width" / Int32un,
            "height" / Int32un,
            "bits_per_pixel" / Int32un,
            "num_slices_0x2" / Int32un,
            "0x1" / Int8un,
            "size" / Int32un,
        )

        self.param_obs_02_header = Struct(
            "camera_model" / PaddedString(12, "utf16"),
            "jpeg_quality" / Int8un,
            "color_temparature" / Int8un,
        )

        self.img_mot_comp_03_header = Struct(
            "width" / Int32un,
            "height" / Int32un,
            "bits_per_pixel" / Int32un,
            "num_slices" / Int32un,
        )

        self.img_projection_header = Struct(
            "width" / Int32un,
            "height" / Int32un,
            "bits_per_pixel" / Int32un,
            "0x1000002" / Int32un,
            "size" / Int32un,
        )

        self.effective_scan_range_header = Struct(
            "bounding_box_fundus_pixel" / Int32un[4],
            "bounding_box_trc_pixel" / Int32un[4],
        )

        self.regist_info_header = Struct(
            "bounding_box_in_fundus_pixels" / Int32un[4],
            "bounding_box_in_trc_pixels" / Int32un[4],
        )

        self.result_cornea_curve_header = Struct(
            "id" / Int8un[20],
            "width" / Int32un,
            "height" / Int32un,
            "8.0.1.21781" / Int8un[32],
        )

        self.result_cornea_thickness_header = Struct(
            "8.0.1.21781" / Int8un[32],
            "id" / Int8un[20],
            "width" / Int32un,
            "height" / Int32un,
        )

        self.contour_info_header = Struct(
            "id" / Int8un[20],
            "type" / Int16un,
            "width" / Int32un,
            "height" / Int32un,
            "size" / Int32un,
        )

        self.align_info_header = Struct (
            "0x0" / Int16un,
            "num_slices" / Int32un,
            "8.0.1.22004" / Int8un[32]
        )

        self.fast_q2_info_header = Struct (
            "various_quality_statistics" / Float32n[6]
        )

        self.gla_littmann_01_header = Struct (
            "0xffff" / Int32un,
            "0x1" / Int32un
        )

        self.chunk_dict = self.get_list_of_file_chunks()

    def get_list_of_file_chunks(self,show =True):
        """Find all data chunks present in the file.

        Returns:
            dict
        """
        chunk_dict = {}
        with open(self.filepath, "rb") as f:
            # skip header
            raw = f.read(15)
            header = self.header.parse(raw)

            eof = False
            while not eof:
                chunk_name_size = np.fromstring(f.read(1), dtype=np.uint8)[0]
                if chunk_name_size == 0:
                    eof = True
                else:
                    chunk_name = f.read(chunk_name_size)
                    chunk_size = np.fromstring(f.read(4), dtype=np.uint32)[0]
                    chunk_location = f.tell()
                    f.seek(chunk_size, 1)
                    chunk_dict[chunk_name] = [chunk_location, chunk_size]
        if show:
            print("File {} contains the following chunks:".format(self.filepath))
            for key in chunk_dict.keys():
                print(key)
            print('')
        return chunk_dict

    def read_oct_volume(self):
        """Reads OCT data.

        Returns:
            obj:OCTVolumeWithMetaData
        """

        if b"@IMG_JPEG" not in self.chunk_dict:
            print("@IMG_JPEG IS NOT IN CHUNKS SKIPPING...")
            return None
        with open(self.filepath, "rb") as f:
            chunk_location, chunk_size = self.chunk_dict[b"@IMG_JPEG"]
            f.seek(chunk_location)  # Set the chunk’s current position.
            raw = f.read(25)
            oct_header = self.oct_header.parse(raw)
            volume = np.zeros(
                (oct_header.height, oct_header.width, oct_header.number_slices)
            )

            for i in range(oct_header.number_slices):
                size = np.fromstring(f.read(4), dtype=np.int32)[0]
                raw_slice = f.read(size)
                image = Image.open(io.BytesIO(raw_slice))
                slice = np.asarray(image)
                volume[:, :, i] = slice

        oct_volume = OCTVolumeWithMetaData(
            [volume[:, :, i] for i in range(volume.shape[2])]
        )
        return oct_volume

    def read_oct_volume_2(self):
        """Reads OCT data.

        Returns:
            obj:OCTVolumeWithMetaData
        """

        if b"@IMG_MOT_COMP_03" not in self.chunk_dict:
            print("@IMG_MOT_COMP_03 IS NOT IN CHUNKS SKIPPING...")
            return None
        with open(self.filepath, "rb") as f:
            chunk_location, chunk_size = self.chunk_dict[b"@IMG_MOT_COMP_03"]
            f.seek(chunk_location)  # Set the chunk’s current position.
            raw = f.read(22)
            oct_header = self.oct_header_2.parse(raw)
            number_pixels = (
                oct_header.width * oct_header.height * oct_header.number_slices
            )
            raw_volume = np.fromstring(f.read(number_pixels * 2), dtype=np.uint16)
            volume = np.array(raw_volume)
            volume = volume.reshape(
                oct_header.width, oct_header.height, oct_header.number_slices, order="F"
            )
            volume = np.transpose(volume, [1, 0, 2])
        oct_volume = OCTVolumeWithMetaData(
            [volume[:, :, i] for i in range(volume.shape[2])]
        )
        return oct_volume

    def read_fundus_image(self):
        """Reads fundus image.

        Returns:
            obj:FundusImageWithMetaData
        """
        if b"@IMG_FUNDUS" not in self.chunk_dict:
            print("@IMG_FUNDUS IS NOT IN CHUNKS SKIPPING...")
            return None
        with open(self.filepath, "rb") as f:
            chunk_location, chunk_size = self.chunk_dict[b"@IMG_FUNDUS"]
            f.seek(chunk_location)  # Set the chunk’s current position.
            raw = f.read(24)  # skip 24 is important
            fundus_header = self.fundus_header.parse(raw)
            number_pixels = fundus_header.width * fundus_header.height * 3
            raw_image = f.read(fundus_header.size)
            image = Image.open(io.BytesIO(raw_image))
            image = np.asarray(image)
        fundus_image = FundusImageWithMetaData(image)
        return fundus_image

    def read_fundus_image_gray_scale(self):
        """Reads gray scale fundus image.

        Returns:
            obj:FundusImageWithMetaData
        """
        if b"@IMG_TRC_02" not in self.chunk_dict:
            print("@IMG_TRC_02 IS NOT IN CHUNKS SKIPPING...")
            return None
        with open(self.filepath, "rb") as f:
            chunk_location, chunk_size = self.chunk_dict[b"@IMG_TRC_02"]
            f.seek(chunk_location)  # Set the chunk’s current position.
            raw = f.read(21)  # skip 21 is important
            img_trc_02_header = self.img_trc_02_header.parse(raw)
            number_pixels = img_trc_02_header.width * img_trc_02_header.height * 1
            raw_image = f.read(img_trc_02_header.size)
            image = Image.open(io.BytesIO(raw_image))
            image = np.asarray(image)
        fundus_gray_scale_image = FundusImageWithMetaData(image)
        return fundus_gray_scale_image

    def read_any_info_and_make_dict(self,chunk_name):
        """
        Reads chunks, get data and make dictionary
        :param chunk_name: name of the chunk which data will be taken.
        Returns:
            dict:Chunk info Data
        """
        if chunk_name not in self.chunk_dict:
            print(f"{chunk_name} IS NOT IN CHUNKS SKIPPING")
            return None
        with open(self.filepath, "rb") as f:
            chunk_location, chunk_size = self.chunk_dict[chunk_name]
            f.seek(chunk_location)  # Set the chunk’s current position.
            raw = f.read()
            header_name = f"{chunk_name.decode().split('@')[-1].lower()}_header"
            chunk_info_header = dict(self.__dict__[header_name].parse(raw))
            chunks_info = dict()
            for idx,key in enumerate(chunk_info_header.keys()):
                if idx == 0:
                    continue
                if type(chunk_info_header[key]) is ListContainer:
                    chunks_info[key] = list(chunk_info_header[key])
                else:
                    chunks_info[key] = chunk_info_header[key]
        return chunks_info