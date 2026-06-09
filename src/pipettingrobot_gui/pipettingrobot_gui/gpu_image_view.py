from __future__ import annotations

import ctypes

from PySide2.QtCore import Qt
from PySide2.QtGui import (
    QColor,
    QImage,
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
    QOpenGLVertexArrayObject,
    QPainter,
)
from PySide2.QtWidgets import QOpenGLWidget


class GpuImageView(QOpenGLWidget):
    VERTEX_SHADER = """
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

    FRAGMENT_SHADER = """
uniform sampler2D frameTex;
varying vec2 v_texcoord;
void main() {
    gl_FragColor = texture2D(frameTex, v_texcoord);
}
"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._placeholder = title
        self._image: QImage | None = None
        self._texture: QOpenGLTexture | None = None
        self._program: QOpenGLShaderProgram | None = None
        self._vbo: QOpenGLBuffer | None = None
        self._vao: QOpenGLVertexArrayObject | None = None
        self._texture_dirty = False
        self.setMinimumHeight(180)
        self.setAutoFillBackground(False)

    def set_placeholder(self, text: str):
        self._placeholder = text
        if self._image is None:
            self.update()

    def set_image(self, image: QImage | None):
        self._image = image
        self._texture_dirty = True
        self.update()

    def clear_image(self):
        self._image = None
        self._texture_dirty = True
        self.update()

    def initializeGL(self):
        self._program = QOpenGLShaderProgram(self)
        self._program.addShaderFromSourceCode(QOpenGLShader.Vertex, self.VERTEX_SHADER)
        self._program.addShaderFromSourceCode(QOpenGLShader.Fragment, self.FRAGMENT_SHADER)
        self._program.link()

        self._vao = QOpenGLVertexArrayObject(self)
        self._vao.create()
        self._vao.bind()

        self._vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self._vbo.create()
        self._vbo.bind()
        self._vbo.setUsagePattern(QOpenGLBuffer.DynamicDraw)

        if self._program is not None:
            self._program.bind()
            stride = 4 * ctypes.sizeof(ctypes.c_float)
            position_loc = self._program.attributeLocation("position")
            texcoord_loc = self._program.attributeLocation("texcoord")
            self._program.enableAttributeArray(position_loc)
            self._program.setAttributeBuffer(position_loc, 0x1406, 0, 2, stride)
            self._program.enableAttributeArray(texcoord_loc)
            self._program.setAttributeBuffer(
                texcoord_loc, 0x1406, 2 * ctypes.sizeof(ctypes.c_float), 2, stride
            )
            self._program.release()

        self._vbo.release()
        self._vao.release()

    def resizeGL(self, width: int, height: int):
        self.context().functions().glViewport(0, 0, width, height)

    def paintGL(self):
        funcs = self.context().functions()
        funcs.glClearColor(22.0 / 255.0, 24.0 / 255.0, 28.0 / 255.0, 1.0)
        funcs.glClear(0x00004000)

        if self._image is None or self._image.isNull():
            self._destroy_texture()
            self._draw_placeholder()
            return

        self._ensure_texture()
        if self._texture is None or self._program is None or self._vbo is None or self._vao is None:
            self._draw_placeholder()
            return

        vertices = self._build_quad_vertices(self._image.width(), self._image.height())
        vertex_bytes = vertices.tobytes()

        self._vao.bind()
        self._vbo.bind()
        self._vbo.allocate(vertex_bytes, len(vertex_bytes))

        self._program.bind()
        self._texture.bind(0)
        self._program.setUniformValue("frameTex", 0)
        funcs.glDrawArrays(0x0005, 0, 4)
        self._texture.release()
        self._program.release()
        self._vbo.release()
        self._vao.release()

    def _ensure_texture(self):
        if not self._texture_dirty:
            return
        self._destroy_texture()

        if self._image is None or self._image.isNull():
            self._texture_dirty = False
            return

        image = self._image.convertToFormat(QImage.Format_RGBA8888).mirrored(False, True)
        self._texture = QOpenGLTexture(image)
        self._texture.setWrapMode(QOpenGLTexture.ClampToEdge)
        self._texture.setMinificationFilter(QOpenGLTexture.Linear)
        self._texture.setMagnificationFilter(QOpenGLTexture.Linear)
        self._texture_dirty = False

    def _destroy_texture(self):
        if self._texture is not None:
            self._texture.destroy()
            self._texture = None

    def _draw_placeholder(self):
        painter = QPainter(self)
        painter.setPen(QColor(201, 209, 217))
        painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
        painter.end()

    def _build_quad_vertices(self, image_width: int, image_height: int):
        import numpy as np

        view_w = max(1, self.width())
        view_h = max(1, self.height())
        image_aspect = float(image_width) / float(max(1, image_height))
        view_aspect = float(view_w) / float(max(1, view_h))

        half_w = 1.0
        half_h = 1.0
        if image_aspect > view_aspect:
            half_h = view_aspect / image_aspect
        else:
            half_w = image_aspect / view_aspect

        return np.array(
            [
                [-half_w, -half_h, 0.0, 0.0],
                [half_w, -half_h, 1.0, 0.0],
                [-half_w, half_h, 0.0, 1.0],
                [half_w, half_h, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

    def closeEvent(self, event):
        self.makeCurrent()
        self._destroy_texture()
        self.doneCurrent()
        super().closeEvent(event)
