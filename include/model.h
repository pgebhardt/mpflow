// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_MODEL_H
#define FASTEIT_INCLUDE_MODEL_H

// namespaces fastEIT
namespace fastEIT {
    // model class definition
    template <
        class BasisFunction
    >
    class Model {
    // constructor and destructor
    public:
        Model(Mesh<BasisFunction>* mesh, Electrodes* electrodes, dtype::real sigmaRef,
            dtype::size numHarmonics, cublasHandle_t handle, cudaStream_t stream);
        virtual ~Model();

    // init methods
    private:
        void init(cublasHandle_t handle, cudaStream_t stream);
        void createSparseMatrices(cublasHandle_t handle, cudaStream_t stream);
        void initExcitationMatrix(cudaStream_t stream);

    public:
        // calc excitaion components
        void calcExcitationComponents(const Matrix<dtype::real>& pattern, cublasHandle_t handle, cudaStream_t stream,
            std::vector<Matrix<dtype::real>*>* components);

        // update model
        void update(const Matrix<dtype::real>& gamma, cublasHandle_t handle, cudaStream_t stream);

    public:
        // accessors
        const Mesh<BasisFunction>& mesh() const { return *this->mesh_; }
        const Electrodes& electrodes() const { return *this->electrodes_; }
        dtype::real sigma_ref() const { return this->sigma_ref_; }
        const std::vector<SparseMatrix*>& system_matrices() const { return this->system_matrices_; }
        const Matrix<dtype::real>& excitation_matrix() const { return *this->excitation_matrix_; }
        dtype::size num_harmonics() const { return this->num_harmonics_; }
        const SparseMatrix& s_matrix() const { return *this->s_matrix_; }
        const SparseMatrix& r_matrix() const { return *this->r_matrix_; }
        const Matrix<dtype::index>& connectivity_matrix() const { return *this->connectivity_matrix_; }
        const Matrix<dtype::real>& elemental_s_matrix() const { return *this->elemental_s_matrix_; }
        const Matrix<dtype::real>& elemental_r_matrix() const { return *this->elemental_r_matrix_; }

        // mutators
        std::vector<SparseMatrix*>& system_matrices() { return this->system_matrices_; }
        Matrix<dtype::real>& excitation_matrix() { return *this->excitation_matrix_; }
        SparseMatrix& s_matrix() { return *this->s_matrix_; }
        SparseMatrix& r_matrix() { return *this->r_matrix_; }
        Matrix<dtype::index>& connectivity_matrix() { return *this->connectivity_matrix_; }
        Matrix<dtype::real>& elemental_s_matrix()  { return *this->elemental_s_matrix_; }
        Matrix<dtype::real>& elemental_r_matrix() { return *this->elemental_r_matrix_; }

    // member
    private:
        Mesh<BasisFunction>* mesh_;
        Electrodes* electrodes_;
        dtype::real sigma_ref_;
        std::vector<SparseMatrix*> system_matrices_;
        SparseMatrix* s_matrix_;
        SparseMatrix* r_matrix_;
        Matrix<dtype::real>* excitation_matrix_;
        Matrix<dtype::index>* connectivity_matrix_;
        Matrix<dtype::real>* elemental_s_matrix_;
        Matrix<dtype::real>* elemental_r_matrix_;
        dtype::size num_harmonics_;
    };
}

#endif
