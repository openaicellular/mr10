# ==================================================================================
#       Copyright (c) 2020 China Mobile Technology (USA) Inc. Intellectual Property.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# ==================================================================================

"""
lp module responsible for SDL queries
"""
import json
#from lp.exceptions import UENotFound

class UENotFound(BaseException):
    pass
class CellNotFound(BaseException):
    pass

print('///////////enter sdl.py//////////')
# namespaces
UE_NS = "TS-UE-metrics"
CELL_NS = "TS-cell-metrics"

# constants
MEASTSRF = "MeasTimestampRF"
MEASPRF = "MeasPeriodRF"
CELLMEAS = "CellMeasurements"

# list of keys in ue metrics
UE_KEY_LIST = set(
    [
        "ServingCellID",
        "MeasTimestampUEPDCPBytes",
        "MeasPeriodUEPDCPBytes",
        "UEPDCPBytesDL",
        "UEPDCPBytesUL",
        "MeasTimestampUEPRBUsage",
        "MeasPeriodUEPRBUsage",
        "UEPRBUsageDL",
        "UEPRBUsageUL",
    ]
)

print('UE_KEY_LIST=', UE_KEY_LIST)
# list of keys in cell metrics
CELL_KEY_LIST = set(
    [
        "CellID",
        "MeasTimestampPDCPBytes",
        "MeasPeriodPDCPBytes",
        "PDCPBytesDL",
        "PDCPBytesUL",
        "MeasTimestampAvailPRB",
        "MeasPeriodAvailPRB",
        "AvailPRBDL",
        "AvailPRBUL",
    ]
)

print('CELL_KEY_LIST=', CELL_KEY_LIST)
def get_uedata(xapp_ref, ueid):
    print('//////////enter def get_uedata in sdl/////////////////')
    """
    this function takes in a single ueid and:
        - fetches the current ue data
        - for the serving cell id, and for each neighboring cell id, fetches the cell data for those cells
        - returns the combined data to lp to perform prediction
    """
    print('sdl_get(UE_NS, ueid, usemsgpack=False)=', sdl_get(UE_NS, ueid, usemsgpack=False))
    ue_data_bytes = xapp_ref.sdl_get(UE_NS, ueid, usemsgpack=False)
    print('ue_data_bytes=', ue_data_bytes)
    if not ue_data_bytes:
        print(' if not ue_data_bytes:')
        raise UENotFound()
    # input should be a json encoded as bytes
    ue_data = json.loads(ue_data_bytes.decode())
    print('ue_data = json.loads(ue_data_bytes.decode())=', ue_data)

    serving_cid = ue_data["ServingCellID"]
    print('serving_cid = ue_data["ServingCellID"]=', serving_cid)

    # a dict is better than a list for what we need to do here
    n_cell_info = {}
    for ncell in ue_data["NeighborCellRF"]:
        print(' for ncell in ue_data["NeighborCellRF"]:')
        n_cell_info[ncell["CID"]] = ncell["CellRF"]
        print('n_cell_info[ncell["CID"]]=', n_cell_info[ncell["CID"]])

    # form the cell_id list
    cell_ids = list(n_cell_info.keys())
    print('cell_ids = list(n_cell_info.keys())=', cell_ids)
    cell_ids.append(serving_cid)
    print('cell_ids.append=', cell_ids)

    # putting together the data from SDL
    lp_data = {"PredictionUE": ueid}  # top level key
    print('lp_data = {"PredictionUE": ueid}=', lp_data)
    lp_data["UEMeasurements"] = {k: ue_data[k] for k in UE_KEY_LIST}  # take ue keys we want
    print('lp_data["UEMeasurements"] = {k: ue_data[k] for k in UE_KEY_LIST} =', lp_data["UEMeasurements"])
    lp_data[CELLMEAS] = []

    # form the Cell Measurements
    for cid in cell_ids:

        cellm_bytes = xapp_ref.sdl_get(CELL_NS, cid, usemsgpack=False)
        print('cellm_bytes=', cellm_bytes)

        if cellm_bytes:  # if None, then we omit that cell from this array

            # input should be a json encoded as bytes
            cellm = json.loads(cellm_bytes.decode())
            print('cellm = json.loads(cellm_bytes.decode())=', cellm)

            # if we were really under performance strain here we could delete
            # from the orig instead of copying but this code is far simpler
            cell_data = {k: cellm[k] for k in CELL_KEY_LIST}
            print('cell_data = {k: cellm[k] for k in CELL_KEY_LIST}=', cell_data)

            # these keys get dropped into *each* cell
            cell_data[MEASTSRF] = ue_data[MEASTSRF]
            print('cell_data[MEASTSRF] = ue_data[MEASTSRF]=', cell_data[MEASTSRF])
            cell_data[MEASPRF] = ue_data[MEASPRF]
            print('cell_data[MEASPRF] = ue_data[MEASPRF]=', cell_data[MEASPRF] )

            # add the RF
            cell_data["RFMeasurements"] = ue_data["ServingCellRF"] if cid == serving_cid else n_cell_info[cid]
            print('cell_data["RFMeasurements"] = ue_data["ServingCellRF"] if=', cell_data["RFMeasurements"])

            # add to our array
            lp_data[CELLMEAS].append(cell_data)
            print('lp_data[CELLMEAS].append(cell_data)=', lp_data[CELLMEAS])

    return lp_data
