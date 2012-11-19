// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_ELECTRODES_HPP
#define FASTEIT_ELECTRODES_HPP

// namespace fastEIT
namespace fastEIT {
    // Electrodes class definition
    class Electrodes {
    // constructer and destructor
    public:
        Electrodes(linalgcuSize_t count, linalgcuMatrixData_t width, linalgcuMatrixData_t height,
            linalgcuMatrixData_t meshRadius);
        virtual ~Electrodes();

    // access methods
    public:
        linalgcuSize_t count() const { return this->mCount; }
        linalgcuMatrixData_t* electrodesStart() const { return this->mElectrodesStart; }
        linalgcuMatrixData_t* electrodesEnd() const { return this->mElectrodesEnd; }
        linalgcuMatrixData_t width() const { return this->mWidth; }
        linalgcuMatrixData_t height() const { return this->mHeight; }

    // member
    private:
        linalgcuSize_t mCount;
        linalgcuMatrixData_t* mElectrodesStart;
        linalgcuMatrixData_t* mElectrodesEnd;
        linalgcuMatrixData_t mWidth;
        linalgcuMatrixData_t mHeight;
    };
}

#endif
