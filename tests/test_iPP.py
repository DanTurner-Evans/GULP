from imganlys import ImagingPreProc as iPP
from pathlib import Path
import datetime
import pytest

class TestPickle:
    tiff_file = Path("Z:/2PImaging/Kerstin/HD7-wtb/20230119_HD7_wtb/tiffs/20230119_6s-ss96-wtb_brain2_r1.tif")
    PROC_DAT_FOLDER = Path("C:/Users/ahshenas/Documents/GitHub/glu-puff-analysis/results/pickle")
    def test_getPicklePath(self):
        pickle_parents = Path(self.PROC_DAT_FOLDER, "2023_01")
        pickle_name = Path('2023-01-19T15:51:41' + self.tiff_file.with_suffix(".pickle").name)
        pickle_path  = Path(pickle_parents, pickle_name)
        assert iPP.getPicklePath(self.tiff_file, self.PROC_DAT_FOLDER) == pickle_path

def test_getDate():
    tiff_file = Path("Z:/2PImaging/Kerstin/HD7-wtb/20230119_HD7_wtb/tiffs/20230119_6s-ss96-wtb_brain2_r1.tif")
    date = iPP.getDate(tiff_file)
    assert date == datetime.datetime(2023, 1, 19, 15, 51, 41)
    assert iPP.formatDate(date.year, date.month)  == "2023_01"