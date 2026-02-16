#!/usr/bin/env bash

set -euo pipefail

CURL_OPTS="-C - --fail --location --progress-bar"

download () {
    local url="$1"
    local out="$2"
    echo "Downloading $out"
    curl $CURL_OPTS "$url" --output "$out"
}

# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_001_150_DCM.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=QfuEUiku2hNiuDNcrljNQP5ozU8%3D&Expires=1776804460" fastMRI_breast_IDS_001_150_DCM.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_150_300_DCM.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=oXjz%2BL8TDzJs901Wpwcl%2BzOp2gA%3D&Expires=1776804460"   fastMRI_breast_IDS_150_300_DCM.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_001_010.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=8H4NAl4YSdfCtyLDCsb%2Bpj6FLp0%3D&Expires=1776804460"   fastMRI_breast_IDS_001_010.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_011_020.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=urdIZMn6k210%2BXuT8kscCd5nnC4%3D&Expires=1776804460"   fastMRI_breast_IDS_011_020.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_021_030.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=7b5I0qE2gTjDfsMrbwvEJVYgXb0%3D&Expires=1776804460"   fastMRI_breast_IDS_021_030.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_031_040.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=oUuyntS7yaJNgQz9x1pDW6mui%2BY%3D&Expires=1776804460"   fastMRI_breast_IDS_031_040.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_041_050.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=jTAZoIo2%2Bx03pcLiH7cvUfJgrHE%3D&Expires=1776804460"   fastMRI_breast_IDS_041_050.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_051_060.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=jNKFKxiks5%2Bsyv07O3AdWZdZGXk%3D&Expires=1776804460"   fastMRI_breast_IDS_051_060.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_061_070.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=y1IjeITJqppQTrAl4zvrJg4mRtc%3D&Expires=1776804460"   fastMRI_breast_IDS_061_070.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_071_080.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=ppmUUb43040tS0bayDBrFDwRgZ0%3D&Expires=1776804460"   fastMRI_breast_IDS_071_080.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_081_090.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=DWtU%2FzTo8zil3hQtbPhx6m8%2BmlM%3D&Expires=1776804460"   fastMRI_breast_IDS_081_090.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_091_100.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=W%2BtfhVZkx3MzFF9hvRXYOqcfeng%3D&Expires=1776804460"   fastMRI_breast_IDS_091_100.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_101_110.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=M90BfQY4v2QczuluSdacVbqpyFk%3D&Expires=1776804460"   fastMRI_breast_IDS_101_110.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_111_120.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=hrC4323OViGrBTwfj%2FkbT0KrZmA%3D&Expires=1776804460"   fastMRI_breast_IDS_111_120.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_121_130.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=vfXwVEylMA%2BT3Q%2BDrWlJDVRpWlo%3D&Expires=1776804460"   fastMRI_breast_IDS_121_130.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_131_140.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=%2BX6lyNLrBujGGWbdXPtQTqBqdGE%3D&Expires=1776804460"   fastMRI_breast_IDS_131_140.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_141_150.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=mEqgY5cUq0FKXNWSOFyVzYnALbc%3D&Expires=1776804460"   fastMRI_breast_IDS_141_150.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_151_160.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=c5kkzLbbWZTHTvi60JXeT3yKYc0%3D&Expires=1776804460"   fastMRI_breast_IDS_151_160.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_161_170.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=jxUdIzwQ8q9wv%2FNkXTrgc61cdt8%3D&Expires=1776804460"   fastMRI_breast_IDS_161_170.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_171_180.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=in89oxvVRgONYfYHzLDhXazCfPw%3D&Expires=1776804460"   fastMRI_breast_IDS_171_180.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_181_190.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=t3HpmMMZf45mVvhU5r9vGXD2PoU%3D&Expires=1776804460"   fastMRI_breast_IDS_181_190.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_191_200.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=LhFV6NZAWcp4pbon2PzVcf9Rk2A%3D&Expires=1776804460"   fastMRI_breast_IDS_191_200.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_201_210.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=PB0AeEXWPyaPkqNEnoMbbkYyYjk%3D&Expires=1776804460"   fastMRI_breast_IDS_201_210.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_211_220.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=vpwh%2FBzNwfINSSTxCdgAdwjR%2F50%3D&Expires=1776804460"   fastMRI_breast_IDS_211_220.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_221_230.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=bcrQNeCIgqC3mUYV0DCflaQ%2FjO4%3D&Expires=1776804460"   fastMRI_breast_IDS_221_230.tar.gz
# download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_231_240.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=uuiXfJ5wbR2lKVIa%2F%2BxS4YAx7Sk%3D&Expires=1776804460"   fastMRI_breast_IDS_231_240.tar.gz
download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_241_250.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=qw3M%2B9eWrsHSur4Nw08XVOzrqEk%3D&Expires=1776804460"   fastMRI_breast_IDS_241_250.tar.gz
download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_251_260.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=oyX5E2FJZElEmhYQs0cqBrVbtII%3D&Expires=1776804460"   fastMRI_breast_IDS_251_260.tar.gz
download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_261_270.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=hCdrDWm9Sk9%2F03sFgLJx2Xc3kVc%3D&Expires=1776804460"   fastMRI_breast_IDS_261_270.tar.gz
download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_271_280.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=FaroR%2Bjl3X4K4j926%2B73g%2FFDWOE%3D&Expires=1776804460"   fastMRI_breast_IDS_271_280.tar.gz
download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_281_290.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=4p31ojuxJ4fGNKk0jGFUWzB9qfk%3D&Expires=1776804460"   fastMRI_breast_IDS_281_290.tar.gz
download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_291_300.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=SqPk%2B%2BcWA%2Fob9HWbxpXEJO1rEXc%3D&Expires=1776804460"   fastMRI_breast_IDS_291_300.tar.gz
download "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_labels.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=Ntxmk%2FA6sOmH57kLGj6GwVCPX18%3D&Expires=1776804460"   fastMRI_breast_labels.tar.gz
download "https://fastmri-dataset.s3.amazonaws.com/v3.0/SHA256?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=opxMzafOxPP3HbUR%2BabtNAJxd8g%3D&Expires=1776804460"   SHA256