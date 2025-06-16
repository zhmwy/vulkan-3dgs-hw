/*
* Basic camera class
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "base/math/matrix.h"
#include "base/math/matrix_util.h"
#include "base/math/quaternion_util.h"
#include "base/math/vector.h"

constexpr double DEG_TO_RAD = M_PI / 180.0;

class Camera
{
private:
	float fov;
	float znear, zfar;
    bool forwad;

    void updateViewMatrix()
	{
		Base::Math::Mat4X4 rotM = Base::Math::Mat4X4(1.0f);
		Base::Math::Mat4X4 transM;

		rotM = rotM * Base::Math::Mat4Cast(Base::Math::AngleAxis(DEG_TO_RAD * rotation.x * (flipY ? -1.0f : 1.0f), Base::Math::Vec3(1.0f, 0.0f, 0.0f)));
		rotM = rotM * Base::Math::Mat4Cast(Base::Math::AngleAxis(DEG_TO_RAD * rotation.y, Base::Math::Vec3(0.0f, 1.0f, 0.0f)));
		rotM = rotM * Base::Math::Mat4Cast(Base::Math::AngleAxis(DEG_TO_RAD * rotation.z, Base::Math::Vec3(0.0f, 0.0f, 1.0f)));

		Base::Math::Vec3 translation(position[0], position[1], position[2]);
		if (flipY) {
			translation.y *= -1.0f;
		}
		transM = Base::Math::Translate(Base::Math::Mat4X4(1.0f), translation);

		if (type == CameraType::firstperson)
		{
			matrices.view = rotM * transM;
		}
		else
		{
			matrices.view = transM * rotM;
		}

		viewPos = Base::Math::Vec4(position, 0.0f) * Base::Math::Vec4(-1.0f, 1.0f, -1.0f, 1.0f);

		updated = true;
	};
public:
	enum CameraType { lookat, firstperson };
	CameraType type = CameraType::lookat;
    
	Base::Math::Vec3 rotation = Base::Math::Vec3();
	Base::Math::Vec3 position = Base::Math::Vec3();
	Base::Math::Vec4 viewPos = Base::Math::Vec4();

	float rotationSpeed = 1.0f;
	float movementSpeed = 1.0f;

	bool updated = false;
	bool flipY = false;

	struct
	{
		Base::Math::Mat4X4 perspective;
		Base::Math::Mat4X4 view;
	} matrices;

	struct
	{
		bool left = false;
		bool right = false;
		bool up = false;
		bool down = false;
	} keys;

	void setPerspective(float fov, float aspect, float znear, float zfar)
	{
		this->fov = fov;
		this->znear = znear;
		this->zfar = zfar;
		matrices.perspective = Base::Math::PerspectiveLhZo(DEG_TO_RAD * fov, aspect, znear, zfar);
		if (flipY) {
			matrices.perspective[1][1] *= -1.0f;
		}
	};

	void setPosition(Base::Math::Vec3 position)
	{
		this->position = position;
		updateViewMatrix();
	}

	void setRotation(Base::Math::Vec3 rotation)
	{
		this->rotation = rotation;
		updateViewMatrix();
	}


};