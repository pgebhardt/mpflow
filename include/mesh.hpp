// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_MESH_HPP
#define FASTEIT_INCLUDE_MESH_HPP

// namespace fastEIT
namespace fastEIT {
    // mesh class definition
    class Mesh {
    // constructor and destructor
    public:
        Mesh(Matrix<dtype::real>& nodes, Matrix<dtype::index>& elements,
            Matrix<dtype::index>& boundary, dtype::real radius, dtype::real height);
        virtual ~Mesh();

    // access methods
    public:
        const Matrix<dtype::real>& nodes() const { return *this->nodes_; }
        const Matrix<dtype::index>& elements() const { return *this->elements_; }
        const Matrix<dtype::index>& boundary() const { return *this->boundary_; }
        dtype::real radius() const { return this->radius_; }
        dtype::real height() const { return this->height_; }

    // member
    private:
        Matrix<dtype::real>* nodes_;
        Matrix<dtype::index>* elements_;
        Matrix<dtype::index>* boundary_;
        dtype::real radius_;
        dtype::real height_;
    };
}

#endif
