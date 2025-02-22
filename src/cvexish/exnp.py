import numpy as np


def central_diff_1d(v: np.ndarray, h: int = 1) -> np.ndarray:
    """
    1D numpy array に対する中心差分を計算する関数。

    Parameters:
        v (numpy.ndarray): 入力配列。
        step (int): 差分のステップ数。

    Returns:
        numpy.ndarray: 中心差分の結果。
    """
    # 中心差分を計算
    dv = v[h:] - v[:-h]

    # 結果配列を初期化
    result = np.zeros_like(v, dtype=v.dtype)

    # 同じ shape で返すために、zero array に差分を挿入
    front = h // 2
    back = h - front
    result[front:-back] = dv

    return result


def central_diff(array: np.ndarray, h: int = 1, axis: int = -1) -> np.ndarray:
    """
    numpy array に対する中心差分を計算する関数。

    Parameters:
        arr (numpy.ndarray): 入力配列（3D）。
        step (int): 差分のステップ数。
        axis (int): 差分を計算する軸。

    Returns:
        numpy.ndarray: 中心差分の結果。
    """
    axis = array.ndim - 1 if axis == -1 else axis

    # 差分のスライスを作成
    slices_from = [slice(None)] * array.ndim
    slices_from[axis] = slice(h, None)

    slices_to = [slice(None)] * array.ndim
    slices_to[axis] = slice(None, -h)

    # 中心差分を計算
    dv = array[tuple(slices_from)] - array[tuple(slices_to)]

    # 結果配列を初期化
    result = np.zeros_like(array, dtype=array.dtype)

    # 計算結果を挿入
    front = h // 2
    back = h - front
    result[tuple(slice(front, -back) if i == axis else slice(None) for i in range(array.ndim))] = dv

    return result
