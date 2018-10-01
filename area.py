import re
import glob
import os.path

import matplotlib.pyplot as plt
import pandas as pd
import numpy
import scipy.linalg
import scipy.integrate
import scipy.stats
import matplotlib.image as mpimg

def extract_photo_index(path):
    basename = os.path.basename(str(path))
    match = re.search(r'^([0-9]+)', basename)
    if match is None:
        return None
    else:
        return int(match.group(1))

    
def read_tps(infile):
    """
    Read a TPS file and return each record as a dictionary
    
    The function returns a list of records. Each records is a
    dictionnary, the keys of which are ``LM{x,y}n`` (with n an index) for
    the landmarks, ``CP{x,y}n`` (with n an index) for the curve points,
    ``nLM`` the number of landmarks, ``nCP`` the number of curve points,
    ``image`` for the path of the initial picture, ``scale`` for the image
    scaling factor, and ``id`` for the id used in the TPS file.
    
    .. warning::
    
        The parser is **not** generalist! It assumes no more than one curve
        is defined.
    
    Parameters
    ----------
    
    infile: iterator on lines
        The TPS file to read as an open file or a list of lines
    
    Returns
    -------
    
    records: list of dict
        A list of records, each record is a dictionary
    
    Raises
    ------
    
    ValueError
        The file contains an unknown key
    """
    re_key_value = re.compile(r'^(?P<key>[A-Z]+)=(?P<value>.*)$')
    records = []
    # The context is useful to identify landmarks and curve points.
    # Indeed, whithout context it is not possible to know if a
    # line containing a set of coordinates corresponds to a landmark
    # or to a curve point. The possible contexts are None, ``LM``, and
    # ``CP``.
    current_context = None
    current_record = {}
    current_index = 0  # This index is used for landmark and curve points
    for line in infile:
        match = re_key_value.match(line)
        if match is not None:
            key = match.group('key')
            value = match.group('value').strip()
            if key == 'LM':
                # New records start with the landmark count, so this
                # line not only changes the context to ``LM`` but also
                # trigger a new record.
                records.append({'nLM': 0, 'nCP': 0})
                current_record = records[-1]
                current_context = 'LM'
            elif key == 'CURVES':
                # We assume there will be only one curve
                assert(value == '1')
            elif key == 'POINTS':
                current_context = 'CP'
            elif key == 'ID':
                current_record['id'] = int(value)
            elif key == 'IMAGE':
                current_record['image'] = value
                current_record['photo_index'] = extract_photo_index(value)
            elif key == 'SCALE':
                # The way the scale is written into the file depends on the
                # user locale. Therefore it can use the coma instead of the
                # dot as a decimal separator. Since no thousand separator is
                # used, it is safe to replace comas by dots.
                value = value.replace(',', '.')
                current_record['scale'] = float(value)
            else:
                # There is an unsuported key in the file. Let's fail!
                raise ValueError(key)
        else:
            x, y = [float(token) for token in line.split()]
            for dimension_name, coordinate in zip('xy', (x, y)):
                index_key = 'n{}'.format(current_context)
                key = '{context}{dimension}{index:02d}'.format(context=current_context,
                                                              dimension=dimension_name,
                                                              index=current_record[index_key])
                current_record[key] = coordinate
            current_record[index_key] += 1
    return records
                

def tps_to_dataframe(infile):
    """
    Read a TPS file and return a pandas dataframe
    
    See :fun:`read_tps` for the limitations and the column
    names.
    
    In addition, this function adds columns with the centroid of the
    landmarks and the centroid of the curves. These columns are named
    ``{LM,CP}_centroid_{x,y}``; see :fun:`add_centroids`.
    
    Parameters
    ----------
    
    infile: iterator on lines
        The TPS file to read as an open file or a list of lines
    
    Returns
    -------
    
    df: pd.DataFrame
    """
    records = read_tps(infile)
    df = pd.DataFrame(records)
    #df = df.set_index('photo_index', drop=False)
    df.index = df['id']
    #add_centroids(df)
    #add_groups(df)
    return df


def record_to_arrays(record):
    """
    Extract landmarks and curve points as coordinate arrays
    
    Parameters
    ----------
    
    record: pd.DataFrame
        A single record

    Returns
    -------
    
    landmarks: numpy array or None
        The landmarks coordinates, None if no landmarks are defined
    curve_points: numpy array or None
        The curve points coordinates, None if no curve points are defined
    """
    n_landmarks = record['nLM']
    n_curve_points = record['nCP']
    if n_landmarks:
        max_landmark_id = n_landmarks - 1
        lmx = record.ix['LMx00':'LMx{:02d}'.format(max_landmark_id)]
        lmy = record.ix['LMy00':'LMy{:02d}'.format(max_landmark_id)]
        landmarks = numpy.array([lmx, lmy]).transpose().astype(float)
    else:
        landmarks = None
    if n_curve_points:
        max_curve_point_id = n_curve_points - 1
        cpx = record.ix['CPx00':'CPx{:02d}'.format(max_curve_point_id)]
        cpy = record.ix['CPy00':'CPy{:02d}'.format(max_curve_point_id)]
        curve_points = numpy.array([cpx, cpy]).transpose().astype(float)
    else:
        curve_points = None
    return landmarks, curve_points


def add_centroids(records):
    """
    Add a column with the landmark and curve centroids
    
    Update the records DataFrame **in place** with a
    ``LM_centroid_x``, ``LM_centroid_y``, ``CP_centroid_x``,
    and ``CP_centroid_y`` columns.
    
    Parameters
    ----------
    
    records: pd.DataFrame
        A records DataFrame, see :fun:`read_tps` for the
        expected column names
    """
    lm_centroids = []
    cp_centroids = []
    for row in records['id']:
        record = records.loc[row,:]
        landmarks, curve_points = record_to_arrays(record)
        if landmarks is not None:
            lm_centroids.append(landmarks.mean(axis=0))
        else:
            lm_centroids.append((None, None))
        if curve_points is not None:
            cp_centroids.append(curve_points.mean(axis=0))
        else:
            cp_centroids.append((None, None))
    records.loc[:, 'LM_centroid_x'] = pd.Series([lm[0] for lm in lm_centroids],
                                         index=records['id'])
    records.loc[:, 'LM_centroid_y'] = pd.Series([lm[1] for lm in lm_centroids],
                                         index=records['id'])
    records.loc[:, 'CP_centroid_x'] = pd.Series([cp[0] for cp in cp_centroids],
                                         index=records['id'])
    records.loc[:, 'CP_centroid_y'] = pd.Series([cp[1] for cp in cp_centroids],
                                         index=records['id'])


def add_groups(records):
    """
    Assign a group to each record
    
    Groups are 'male', 'female', and 'juvenile'. The assignation is
    done based on the photo index:
    
    * females are photo indices lesser or equal to 76
    * juveniles are photo indices between 77 and 195, included
    * males are photo indices greater than 195
    """
    groups = []
    for row in records['id']:
        photo_index = records.loc[row, 'photo_index']
        if pd.isnull(photo_index):
            groups.append(None)
        elif photo_index <= 76:
            groups.append('female')
        elif photo_index <= 195:
            groups.append('juvenile')
        else:
            groups.append('male')
    records['group'] = pd.Series(groups, index=records['id'])


def convert_path(path):
    return os.path.join(*path.split('\\')[5:])

def get_angle(vecA, vecB):
    normA = numpy.linalg.norm(vecA)
    normB = numpy.linalg.norm(vecB)
    return numpy.arccos(numpy.dot(vecA, vecB) / (normA * normB))

def get_angle(vecA, vecB):
    cosang = numpy.dot(vecA, vecB)
    sinang = numpy.linalg.norm(numpy.cross(vecA, vecB))
    angle = numpy.arctan2(sinang, cosang)
    #if sinang < 0:
    #    angle *= -1
    return angle

def align_base(curve_points, reference=numpy.array([1, 0])):
    vector = curve_points[-1, :] - curve_points[0, :]
    angle = get_angle(vector, reference)
    
    translation = curve_points[0, :]
    curve_points -= translation
    if curve_points[-1, 1] < curve_points[0, 1]:
        angle *= -1
    rotation_matrix = numpy.array([[numpy.cos(angle), -numpy.sin(angle)],
                                   [numpy.sin(angle),  numpy.cos(angle)]])
    curve_points = curve_points.dot(rotation_matrix)
    curve_points += translation
    
    return curve_points



def get_areas(records):
    areas_simps = []
    areas_trapz = []
    for row in records['id']:
        lm, cp = record_to_arrays(records.loc[row, :])
        cp = align_base(cp)
        x = cp[:, 0]
        y = cp[:, 1]
        areas_simps.append(scipy.integrate.simps(y, x))
        areas_trapz.append(scipy.integrate.cumtrapz(y, x)[-1])
    records['area_simps'] = pd.Series(areas_simps, index=records['id'])
    records['area_trapz'] = pd.Series(areas_trapz, index=records['id'])
    records['area_simps_scaled'] = records['area_simps'] * records['scale'] ** 2
    records['area_trapz_scaled'] = records['area_trapz'] * records['scale'] ** 2

def distance(pointA, pointB):
    return numpy.sum((pointA - pointB) ** 2, axis=1) ** 0.5


def perimeter(curve_points):
    return distance(curve_points[:-1, :], curve_points[1:, :]).sum()


def add_perimeter(records):
    perimeters = []
    for row in records['id']:
        lm, cp = record_to_arrays(records.loc[row, :])
        perimeters.append(perimeter(cp))
    records['perimeter'] = pd.Series(perimeters, index=records['id'])
    records['perimeter_scaled'] = records['perimeter'] * records['scale']

    
def draw_one(axis, path, main, other, scale,
             only_crest=True, marker_main=None, marker_other=None):
    # Draw the picture
    img = mpimg.imread(path)
    img = img[::-1, :]
    extent = [0, img.shape[1] * scale,
              0, img.shape[0] * scale]
    img = axis.imshow(img, origin='lower', extent=extent)
    
    # Draw main
    axis.plot(main[:, 0], main[:, 1], color='k', lw=2, marker=marker_main)
    
    # Align other
    vec = main[-1, :] - main[0, :]
    mod = align_base(other.copy(), vec) + main[0, :]
    axis.plot(mod[:, 0], mod[:, 1], color='k', lw=2, ls='--', marker=marker_other)
    
    # Zoom on the crest
    if only_crest:
        max_x = max(main[:, 0].max(), mod[:, 0].max())
        min_x = min(main[:, 0].min(), mod[:, 0].min())
        max_y = max(main[:, 1].max(), mod[:, 1].max())
        min_y = min(main[:, 1].min(), mod[:, 1].min())
        margin_x = 0.2 * (max_x - min_x)
        margin_y = 0.2 * (max_y - min_y)
        axis.set_xlim(min_x - margin_x, max_x + margin_x)
        axis.set_ylim(min_y - margin_y, max_y + margin_y)