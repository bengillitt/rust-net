pub fn matrix_multiply_vector(mat_1: Vec<Vec<f32>>, mat_2: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>, String> {
    if mat_1.len() == 0 || mat_2.len() == 0 {
        return Ok(vec![]);
    }

    if mat_1[0].len() != mat_2.len() {
        return Err("Matrix orders don't match".to_string());
    }

    let mut mat_out: Vec<Vec<f32>> = vec![];

    for i in 0..mat_1.len() {
        let mut row_out: Vec<f32> = vec![];
        for j in 0..mat_2[0].len() {
            let mut current: f32 = 0.0;

            for k in 0..mat_1[0].len() {
                current += mat_1[i][k] * mat_2[k][j];
            }

            row_out.push(current);
        }

        mat_out.push(row_out);
    }

    return Ok(mat_out);
}

#[derive(Debug, Clone)]
pub struct FlatMatrix {
    pub mat: Vec<f32>,
    pub rows: usize,
}

impl FlatMatrix {
    pub fn cols(&self) -> usize {
        return self.mat.len() / self.rows;
    }
}

#[inline(never)]
pub fn matrix_multiply_flat(mat_1: &FlatMatrix, mat_2: &FlatMatrix) -> Result<FlatMatrix, String> {
    if mat_1.mat.len() % mat_1.rows != 0 || mat_2.mat.len() % mat_2.rows != 0 {
        return Err("Rows don't have same length".to_string());
    }

    // Rows = m, cols = n

    let mat_1_cols = mat_1.cols();
    let mat_2_cols = mat_2.cols();

    // Check middle orders match
    if mat_1_cols != mat_2.rows {
        return Err("Matrix orders don't match".to_string());
    }

    let out_size = mat_1.rows * mat_2_cols;

    let mut mat_out = FlatMatrix{mat: vec![0.0; out_size], rows: mat_1.rows};

    for i in 0..mat_1.rows {
        let row_1_offset = i * mat_1_cols;
        let out_row = &mut mat_out.mat[i * mat_2_cols..(i + 1) * mat_2_cols];

        for k in 0..mat_1_cols {
            let val_1 = mat_1.mat[row_1_offset + k];
            let row_2 = &mat_2.mat[k * mat_2_cols .. (k + 1) * mat_2_cols];

            for j in 0..mat_2_cols {
                out_row[j] += val_1 * row_2[j];
            }
        }
    }

    return Ok(mat_out);
}

pub fn transpose_mat(mat: FlatMatrix) -> Result<FlatMatrix, String> {
    if mat.mat.len() % mat.rows != 0 || mat.rows == 0{
        return Err("Invalid matrix dimensions".to_string());
    }

    let cols = mat.cols();
    let rows = mat.rows;

    let mut t_mat = vec![0.0; mat.mat.len()];

    const BLOCK_SIZE: usize = 32;

for r_block in (0..rows).step_by(BLOCK_SIZE) {
        for c_block in (0..cols).step_by(BLOCK_SIZE) {
            
            let r_end = (r_block + BLOCK_SIZE).min(rows);
            let c_end = (c_block + BLOCK_SIZE).min(cols);

            for r in r_block..r_end {
                let in_row_offset = r * cols;
                for c in c_block..c_end {
                    // Transpose mapping: out[c, r] = in[r, c]
                    t_mat[c * rows + r] = mat.mat[in_row_offset + c];
                }
            }
        }
    }

    Ok(FlatMatrix { mat: t_mat, rows: cols })
}