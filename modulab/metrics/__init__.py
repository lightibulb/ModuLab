from .quality import snr_db, si_sdr_db, mcd_db
from .cost import wall_time_ms

REGISTRY = {"snr_db": snr_db, "si_sdr_db": si_sdr_db, "mcd_db": mcd_db}
