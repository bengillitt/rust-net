use super::neural_network::layer::ActivationFunction;

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

pub fn matrix_softmax(mat: &FlatMatrix) -> FlatMatrix {
    let cols = mat.cols();

    let mut out = vec![0.0; cols*mat.rows];

    for i in 0..mat.rows {
        let slice = &mat.mat[i*cols..(i+1)*cols];
        let mut sum: f32 = 0.0;

        let out_offset = i*cols;

        let max = slice
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        for j in 0..cols {
            let exp = (slice[j] - max).exp();

            out[out_offset + j] = exp;
            sum += exp;
        }

        for j in 0..cols {
            out[out_offset + j] /= sum;
        }
    }

    return FlatMatrix{mat: out, rows: mat.rows};
}

pub fn matrix_activation(mat: &FlatMatrix, activation: &ActivationFunction) -> FlatMatrix {
    match activation {
        ActivationFunction::Linear => mat.clone(),
        ActivationFunction::ReLU => matrix_activate_relu(mat),
        ActivationFunction::Sigmoid => matrix_activate_sigmoid(mat),
        ActivationFunction::Tanh => matrix_activate_tanh(mat),
    }
}

fn matrix_activate_relu(mat: &FlatMatrix) -> FlatMatrix {
    let out = mat.mat.iter().map(|&z| if z > 0.0 { z } else { 0.0 } ).collect();

    return FlatMatrix{mat: out, rows: mat.rows};
}

fn matrix_activate_sigmoid(mat: &FlatMatrix) -> FlatMatrix {
    let out = mat.mat.iter().map(|&z| (1.0 / (1.0 + f64::exp(-(z as f64)))) as f32).collect();

    return FlatMatrix{mat: out, rows: mat.rows};
}

fn matrix_activate_tanh(mat: &FlatMatrix) -> FlatMatrix {
    let out = mat.mat.iter().map(|&z| z.tanh()).collect();

    return FlatMatrix{mat: out, rows: mat.rows};
}

pub fn matrix_add_bias_flat(mat: &FlatMatrix, bias: &FlatMatrix) -> Result<FlatMatrix, String> {
    if mat.cols() != bias.cols() || bias.rows != 1 {
        return Err("Incorrect Matrix Order".to_string());
    }

    let cols = mat.cols();

    let out = mat
        .mat
        .chunks_exact(cols)
        .flat_map(|row| row.iter().zip(bias.mat.iter()).map(|(&x, &b)| x + b))
        .collect();

    return Ok(FlatMatrix{mat: out, rows: mat.rows});
}

pub fn matrix_add_flat(mat_1: &FlatMatrix, mat_2: &FlatMatrix) -> Result<FlatMatrix, String> {
    if mat_1.rows != mat_2.rows || mat_1.cols() != mat_2.cols() {
        return Err("Matrix orders don't match".to_string());
    }

    let mat = mat_1
        .mat
        .iter()
        .zip(mat_2.mat.iter())
        .map(|(&a, &b)| a + b)
        .collect();

    Ok(FlatMatrix { mat, rows: mat_1.rows })
}

#[inline(never)]
pub fn matrix_multiply_flat(mat_1: &FlatMatrix, mat_2: &FlatMatrix, transpose: bool) -> Result<FlatMatrix, String> {
    if mat_1.rows == 0 || mat_2.rows == 0 {
        return Err("Matrix can't have zero rows".to_string());
    }

    if mat_1.mat.len() % mat_1.rows != 0 || mat_2.mat.len() % mat_2.rows != 0 {
        return Err("Invalid Matrix Dimensions".to_string());
    }

    // Rows = m, cols = n

    let mat_1_cols = mat_1.cols();
    let mat_2_cols = mat_2.cols();

    if transpose {
        if mat_1_cols != mat_2_cols {
            return Err("Columns don't match".to_string());
        }

        let mut out = vec![0.0; mat_1.rows * mat_2.rows];

        for i in 0..mat_1.rows {
            let out_offset = i * mat_2.rows;
            let mat_1_offset = mat_1_cols * i;
            for j in 0..mat_2.rows {
                let mat_2_offset = mat_2_cols*j;
                let mut total = 0.0;
                for k in 0..mat_1_cols {
                    total += mat_1.mat[mat_1_offset + k] * mat_2.mat[mat_2_offset + k];
                }

                out[out_offset + j] = total;
            }
        }

        return Ok(FlatMatrix { mat: out, rows: mat_1.rows });
    }

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