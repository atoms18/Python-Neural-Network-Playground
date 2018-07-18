
import wx
from PIL import Image


class OutputPanel(wx.Panel):

    def __init__(self, parent, frame, size, density, **kw):
        self.frame = frame

        self.output_panel_size = (size, size)

        super(OutputPanel, self).__init__(
            parent, size=self.output_panel_size, **kw)

        self.output_animation = wx.StaticBitmap(self)
        self.output_canvas_image = wx.Image(self.output_panel_size)
        self.output_density_image = Image.new('RGBA', (density, density))

    def updateOutputCanvasImage(self, datas):

        self.output_density_image.putdata(datas)
        resize_image = self.output_density_image.resize(self.output_panel_size)

        self.output_canvas_image.SetData(
            resize_image.convert("RGB").tobytes())
        self.output_canvas_image.SetAlpha(
            resize_image.convert("RGBA").tobytes()[3::4])

        self.output_animation.SetBitmap(wx.Bitmap(self.output_canvas_image))
