import os
import sys

from ament_index_python.packages import get_package_share_directory
from PySide2.Qt3DCore import Qt3DCore
from PySide2.Qt3DExtras import Qt3DExtras
from PySide2.Qt3DRender import Qt3DRender
from PySide2.QtCore import QTimer, QUrl
from PySide2.QtGui import QColor, QVector3D
from PySide2.QtWidgets import QApplication, QMainWindow, QWidget


def configure_qt_environment():
    cv2_plugin_markers = ('cv2/qt/plugins', 'opencv_python.libs')
    for env_name in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_PLUGIN_PATH'):
        value = os.environ.get(env_name)
        if value and any(marker in value for marker in cv2_plugin_markers):
            os.environ.pop(env_name, None)


def package_uri_to_path(uri: str) -> str:
    if uri.startswith('package://'):
        body = uri[len('package://') :]
        package_name, relative_path = body.split('/', 1)
        return f"{get_package_share_directory(package_name)}/{relative_path}"
    return uri


class DaeViewerWindow(QMainWindow):
    def __init__(self, mesh_path: str):
        super().__init__()
        self.setWindowTitle('Qt3D DAE Viewer Test')
        self.resize(1280, 860)

        self.window = Qt3DExtras.Qt3DWindow()
        self.window.defaultFrameGraph().setClearColor(QColor(28, 30, 34))
        self.root = Qt3DCore.QEntity()
        self.window.setRootEntity(self.root)

        self.camera = self.window.camera()
        self.camera.lens().setPerspectiveProjection(45.0, 16.0 / 9.0, 0.01, 100.0)
        self.camera.setPosition(QVector3D(1.4, -1.6, 1.1))
        self.camera.setViewCenter(QVector3D(0.0, 0.0, 0.2))

        controller = Qt3DExtras.QOrbitCameraController(self.root)
        controller.setCamera(self.camera)
        controller.setLinearSpeed(40.0)
        controller.setLookSpeed(180.0)
        self.camera_controller = controller

        self._build_scene(mesh_path)

        container = QWidget.createWindowContainer(self.window, self)
        self.setCentralWidget(container)

    def _build_scene(self, mesh_path: str):
        ground = Qt3DCore.QEntity(self.root)
        ground_mesh = Qt3DExtras.QPlaneMesh()
        ground_mesh.setWidth(2.0)
        ground_mesh.setHeight(2.0)
        ground_material = Qt3DExtras.QPhongMaterial(ground)
        ground_material.setDiffuse(QColor(70, 74, 82))
        ground_transform = Qt3DCore.QTransform()
        ground_transform.setTranslation(QVector3D(0.0, 0.0, 0.0))
        ground.addComponent(ground_mesh)
        ground.addComponent(ground_material)
        ground.addComponent(ground_transform)

        for position, color in (
            (QVector3D(1.8, -1.5, 2.6), QColor(255, 255, 245)),
            (QVector3D(-1.0, 1.2, 1.5), QColor(170, 195, 255)),
        ):
            light_entity = Qt3DCore.QEntity(self.root)
            light = Qt3DRender.QPointLight(light_entity)
            light.setColor(color)
            light.setIntensity(1.4)
            transform = Qt3DCore.QTransform()
            transform.setTranslation(position)
            light_entity.addComponent(light)
            light_entity.addComponent(transform)

        axis_specs = (
            (QVector3D(0.35, 0.01, 0.01), QVector3D(0.175, 0.0, 0.01), QColor(220, 80, 80)),
            (QVector3D(0.01, 0.35, 0.01), QVector3D(0.0, 0.175, 0.01), QColor(80, 210, 110)),
            (QVector3D(0.01, 0.01, 0.35), QVector3D(0.0, 0.0, 0.175), QColor(80, 130, 230)),
        )
        for size, translation, color in axis_specs:
            axis = Qt3DCore.QEntity(self.root)
            axis_mesh = Qt3DExtras.QCuboidMesh()
            axis_mesh.setXExtent(size.x())
            axis_mesh.setYExtent(size.y())
            axis_mesh.setZExtent(size.z())
            axis_material = Qt3DExtras.QPhongMaterial(axis)
            axis_material.setDiffuse(color)
            axis_transform = Qt3DCore.QTransform()
            axis_transform.setTranslation(translation)
            axis.addComponent(axis_mesh)
            axis.addComponent(axis_material)
            axis.addComponent(axis_transform)

        self.model_root = Qt3DCore.QEntity(self.root)
        self.scene_loader = Qt3DRender.QSceneLoader(self.model_root)
        self.scene_loader.setSource(QUrl.fromLocalFile(mesh_path))
        self.model_root.addComponent(self.scene_loader)
        self.model_transform = Qt3DCore.QTransform()
        self.model_transform.setTranslation(QVector3D(0.0, 0.0, 0.02))
        self.model_transform.setScale3D(QVector3D(1.0, 1.0, 1.0))
        self.model_root.addComponent(self.model_transform)

        fallback = Qt3DCore.QEntity(self.root)
        fallback_mesh = Qt3DExtras.QCuboidMesh()
        fallback_mesh.setXExtent(0.18)
        fallback_mesh.setYExtent(0.12)
        fallback_mesh.setZExtent(0.18)
        fallback_material = Qt3DExtras.QPhongAlphaMaterial(fallback)
        fallback_material.setDiffuse(QColor(255, 170, 90, 120))
        fallback_transform = Qt3DCore.QTransform()
        fallback_transform.setTranslation(QVector3D(0.0, 0.0, 0.09))
        fallback.addComponent(fallback_mesh)
        fallback.addComponent(fallback_material)
        fallback.addComponent(fallback_transform)
        self.fallback_entity = fallback

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._report_loader_status)
        self.status_timer.start(1000)

    def _report_loader_status(self):
        status = self.scene_loader.status()
        if status == Qt3DRender.QSceneLoader.Ready:
            print('DAE scene loader status: Ready')
            self.fallback_entity.setEnabled(False)
            self.status_timer.stop()
        elif status == Qt3DRender.QSceneLoader.Error:
            print('DAE scene loader status: Error')
            self.status_timer.stop()


def resolve_mesh_argument() -> str:
    default_uri = 'package://pipettingrobot_gui/meshes/TestTubeRack.dae'
    mesh_arg = sys.argv[1] if len(sys.argv) > 1 else default_uri
    mesh_path = package_uri_to_path(mesh_arg)
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f'Mesh path not found: {mesh_path}')
    return mesh_path


def main():
    configure_qt_environment()
    mesh_path = resolve_mesh_argument()
    app = QApplication(sys.argv)
    window = DaeViewerWindow(mesh_path)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
