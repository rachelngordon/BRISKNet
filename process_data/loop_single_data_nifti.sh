#/bin/bin -l

# to run this code use the following command: bash loop_single_dir.sh [DATA_DIR] [OUT_DIR] [SCAN_NUM] [CS_MAPS] [SPOKES_PER_FRAME] [SLICE_IDX] [NUM_SLICES] [PER_SPOKES]
# ex: bash loop_single_data_nifti.sh /ess/scratch/scratch1/rachelgordon/fastMRI_breast_data/fastMRI_breast_IDS_001_010/fastMRI_breast_001_1.h5 /ess/scratch/scratch1/rachelgordon/pre-contrast-1tf 1 True 288 0 192 100

# Check if DIR and OUT_DIR are provided as arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <PATH> <OUT_DIR> <SCAN_NUM> [SPOKES] [SLICE_IDX] [SLICE_INC] [PER_SPOKES]"
    exit 1
fi

# Set DIR and OUT_DIR to the provided arguments
DIR="$1"
OUT_DIR="$2"
SCAN_NUM="$3"
CS_MAPS="$4"
KEEP_COMPLEX="$9"




# if [ -z "$1" ]; then
#     echo "Find raw .h5 files in the current directory: $(pwd)"
#     DIR = "fastMRI_breast_001_1"
# else
#     echo "Find raw .h5 files in: $1"
#     DIR="$1"
# fi

#DATA=$(find "${DIR}" -maxdepth 1 -type f -name ${DIR}".h5" -exec basename {} \;)

if [ -z "$5" ]; then
    echo "> SPOKES set as default: 72"
    SPOKES=72
else
    NUM='^[0-9]+$'

    if [[ $5 =~ $NUM ]]; then
        SPOKES="$5"
    else
        echo "> Input $5 is not a number. SPOKES set as default: 72"
        SPOKES=72
    fi
fi

if [ -z "$6" ]; then
    echo "> SLICE_IDX set as default: 0"
    SLICE_IDX=0
else
    NUM='^[0-9]+$'

    if [[ $6 =~ $NUM ]]; then
        SLICE_IDX="$6"
    else
        echo "> Input $6 is not a number. SLICE_IDX set as default: 0"
        SLICE_IDX=0
    fi
fi

if [ -z "$7" ]; then
    echo "> SLICE_INC set as default: 192"
    SLICE_INC=192
else
    NUM='^[0-9]+$'

    if [[ $7 =~ $NUM ]]; then
        SLICE_INC="$7"
    else
        echo "> Input $7 is not a number. SLICE_INC set as default: 192"
        SLICE_INC=192
    fi
fi

if [ -z "$8" ]; then
    echo "> PER_SPOKES set as default: 100"
    PER_SPOKES=100
else
    NUM='^[0-9]+$'
    if [[ $8 =~ $NUM ]]; then
        if [ "$8" -ge 0 ] && [ "$8" -le 100 ]; then
            PER_SPOKES="$8"
        else
            echo "> Input $8 is not a valid percentage (0-100). PER_SPOKES set as default: 100"
            PER_SPOKES=100
        fi
    else
        echo "> Input $8 is not a number. PER_SPOKES set as default: 100"
        PER_SPOKES=100
    fi
fi

    # Extract the basename of the file
    BASEDIR=$(dirname "$DIR")
    DATA=$(basename "$DIR")



    echo "> DIR: ${BASEDIR}"
    echo "> OUT_DIR: ${OUT_DIR}"
    echo "> SCAN_NUM: ${SCAN_NUM}"
    echo "> CS_MAPS: ${CS_MAPS}"
    echo "> DATA: ${DATA}"
    echo "> SPOKES: ${SPOKES}"
    echo "> SLICE_IDX: ${SLICE_IDX}"
    echo "> SLICE_INC: ${SLICE_INC}"
    echo "> PER_SPOKES: ${PER_SPOKES}"
    echo "> KEEP_COMPLEX: ${KEEP_COMPLEX}"

    # Check if SLICE_IDX is equal to zero
    if [ "$SLICE_IDX" -ne 0 ]; then  
        ((SLICE_IDX--))
        echo "SLICE_IDX is now $SLICE_IDX"
    fi 
    # reconstruct slice by slice
    python dce_recon.py --dir "${BASEDIR}" --out_dir "${OUT_DIR}" --data "${DATA}" --save_cs_maps "${CS_MAPS}" --spokes_per_frame "${SPOKES}" --slice_idx "${SLICE_IDX}" --slice_inc "${SLICE_INC}" --per_spokes "${PER_SPOKES}" --images_per_slab "${SLICE_INC}" --keep_complex "${KEEP_COMPLEX}"

    # convert the .h5 file to nifti
    FN="${BASEDIR}/${DATA%.*}"
    echo "> FN: ${FN}"
    echo "> DIR: ${DIR}"
    echo "> DATA: ${DATA}"
    #> FN: /ess/scratch/scratch1/rachelgordon/fastMRI_breast_data/fastMRI_breast_IDS_271_280/fastMRI_breast_279_2.h5/fastMRI_breast_279_2
    python nifti_recon.py --dir "${BASEDIR}" --h5py "${FN}" --spokes_per_frame "${SPOKES}" --out_dir "${OUT_DIR}" --keep_complex "${KEEP_COMPLEX}"