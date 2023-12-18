from math import sqrt
import numpy as np
from numpy.typing import NDArray

# from ._utils import _normalize, _shape_check, _rayleigh_quotient

_EPS = 1e-9

def _rayleigh_quotient(matrix: NDArray[np.float64], vector: NDArray[np.float64]):
    """Находит коэффициент Рэлея для заданных матрицы и вектора."""
    return vector.transpose() @ matrix @ vector / (vector.transpose() @ vector)


def _shape_check(matrix: NDArray[np.float64]) -> None:
    """Проверяет матрицу на квадратность."""
    shape = matrix.shape
    if not (len(shape) == 2 and shape[0] == shape[1]):
        raise ValueError("Input matrix should be square!")


def _normalize(vector: NDArray[np.float64], norm=2) -> NDArray[np.float64]:
    """Нормализует входной вектор."""
    return vector / np.linalg.norm(vector, ord=norm)


def qr_decompose(matrix: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Находит декомпозицию исходной квадратной матрицы на ортогональную и верхнюю треугольную матрицы.
    В данной реализации используется процесс Грамма-Шмидта.
    """
    _shape_check(matrix)  # проверка формы матрицы - если не квадратная будет исключение

    n = matrix.shape[0]
    # Q - ортогональная матрица, R - верхняя треугольная матрица
    q_matrix, r_matrix = np.empty((n, n)), np.empty((n, n))
    # Итерация по к-ву столбцов матрицы:
    for i in range(n):
        # считаем u(n+1) = a(n+1) - r(n+1)
        u_vector = matrix[:, i].astype(float)
        for j in range(i):
            # r(n+1) = (a(n+1) * e1) * e1 + ... + (a(n+1) * ek) * ek
            a_n = (matrix[:, i] @ q_matrix[:, j]) * q_matrix[:, j]
            u_vector -= a_n
        # считаем en = un / ||un||
        q_matrix[:, i] = _normalize(u_vector)

    # Составляем верхнюю треугольную матрицу R
    r_matrix = q_matrix.T @ matrix
    r_matrix = np.triu(np.ones((n, n))) * r_matrix

    return q_matrix, r_matrix


def get_eigenvalues_of_little_matrix(matrix: NDArray[np.float64]) -> tuple[np.float64, np.float64]:
    if matrix.shape != (2, 2):
        raise ValueError('Matrix must be 2x2 size')

    ((a, b), (c, d)) = matrix
    discr = (a - d) ** 2 + 4 * b * c
    discr = 0 if discr < 0 else discr
    num = a + d
    den = 2

    l1, l2 = (num + sqrt(discr)) / den, (num - sqrt(discr)) / den

    return l1, l2


def _get_shift(matrix: NDArray[np.float64], shift_type) -> np.float64:
    shift = None
    if shift_type is None:
        shift = np.float64(0.0)
    elif shift_type == 'Wilkinson':
        eigenvalues = get_eigenvalues_of_little_matrix(matrix[-2:, -2:])
        corner = matrix[-1, -1]
        if abs(eigenvalues[0] - corner) < abs(eigenvalues[1] - corner):
            shift = eigenvalues[0]
        else:
            shift = eigenvalues[1]
    elif shift_type == 'Rayleigh':
        shift = matrix[-1, -1]
    return np.float64(shift)


def qr_method(matrix: NDArray[np.float64], *, iters=50,
              shift_type=None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Находит собственные числа и векторы заданной матрицы."""

    n = matrix.shape[0]
    I = np.eye(n)
    shift = np.float64(0.0)
    eigen_values = matrix.copy()
    eigen_vectors = np.eye(n)

    # Делаем декомпозицию заданное число итераций
    for _ in range(iters):

        q_matrix, r_matrix = qr_decompose(eigen_values - shift * I)
        eigen_values = r_matrix @ q_matrix + shift * I
        eigen_vectors = eigen_vectors @ q_matrix
        shift = _get_shift(eigen_values, shift_type)

    # Диагональ результирующей матрицы - собственные числа исходной матрицы
    # Единично-диагональная матрица, умноженная на
    # ортогональную составляющую разложения в количество итераций - собственные векторы
    return np.diag(eigen_values), eigen_vectors


def rotation_method(matrix: NDArray[np.float64],
                    eps: float = 1e-3)-> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Метод вращений для нахождения собственных векторов и значений матрицы."""
    _shape_check(matrix)

    A = matrix.copy().astype(float)
    max_element = float('inf')
    n = A.shape[0]
    eigenvectors = np.eye(n)

    while abs(max_element) > eps:
        A_masked = A.copy()
        A_masked[np.tril_indices(n, k=0)] = 0
        i, j = np.unravel_index(np.argmax(abs(A_masked)), A.shape)
        max_element = A[i, j]
        H = np.eye(n)

        phi = (0.5 * np.arctan(2 * max_element / (A[i, i] - A[j, j])) if A[i, i] - A[j, j] != 0
               else np.pi / 4)
        H[i, i] = np.cos(phi)
        H[i, j] = -1 * np.sin(phi)
        H[j, i] = np.sin(phi)
        H[j, j] = np.cos(phi)

        A = H.T @ A @ H
        eigenvectors = eigenvectors @ H

    return np.diag(A), eigenvectors


def check_correctness_of_eigen(matrix: NDArray[np.float64], eigenvalues: NDArray[np.float64],
                               eigenvectors: NDArray[np.float64], delta: float = 1e-3) -> bool:
    for value, vector in zip(eigenvalues, eigenvectors.T):
        output_vector = matrix @ vector
        difference = value * vector - output_vector
        if np.linalg.norm(difference, ord=2) > delta:
            return False
    return True


def power_iterations_method(matrix: NDArray[np.float64], *, mu: float=0.0,
                     eps: float=_EPS, inverse: bool=False) -> tuple[np.float64, NDArray[np.float64]]:
    """Находит наибольшее/наименьшее собственное число и соответсвующий
    собственный вектор матрицы методом степенных итераций.

    Обратные итерации
    -----
    Для поиска наименьшей собственной пары использовать аргумент inverse=True.

    Опционально для инвертированных итераций можно указать коэф. мю -
    примерное приближение к какому-то конкретному собственному числу;
    В таком случае получается метод обратных итераций со сдвигом/метод итераций Рэлея
    """
    _shape_check(matrix)

    n = matrix.shape[0]
    eigenvalue_tmp = np.inf

    if inverse:
        # Для обратного метода скопируем исходную матрицу
        # Потом используем ее в коэф. Рэлея, чтобы найти собственное число
        A = np.copy(matrix)
        I = np.eye(n)
        matrix = np.linalg.inv(matrix - mu * I)

    # Найдем начальное приближение
    # Примем начальный вектор [1, ..., 1]
    eigenvector = matrix @ np.ones(n)
    # Собственное число - норма вектора
    eigenvalue = np.linalg.norm(eigenvector)
    # Нормируем вектор
    eigenvector /= eigenvalue

    while np.abs(eigenvalue - eigenvalue_tmp) > eps:
        eigenvalue_tmp = eigenvalue
        # Аналогично начальному приближению
        # Считаем вектор = матрица * прошлый_вектор
        eigenvector = matrix @ eigenvector
        # Собственное число - норма вектора
        eigenvalue = np.linalg.norm(eigenvector)
        # Нормируем вектор
        eigenvector /= eigenvalue

    # Для обратной итерации расчитаем собственное число с помощью коэф. Рэлея
    if inverse:
        eigenvalue = _rayleigh_quotient(A, eigenvector)

    return eigenvalue, eigenvector
