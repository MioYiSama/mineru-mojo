"""Owning bfloat16 matrices shared by the native model modules."""
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

comptime BF16 = DType.bfloat16
comptime MatrixLayout = type_of(row_major(Int32(0), Int32(0)))


@fieldwise_init
struct Tensor(Movable):
    var buffer: DeviceBuffer[BF16]
    var rows: Int
    var cols: Int

    @staticmethod
    def empty(ctx: DeviceContext, rows: Int, cols: Int) raises -> Tensor:
        return Tensor(ctx.enqueue_create_buffer[BF16](rows * cols), rows, cols)

    def reshape(deinit self, rows: Int, cols: Int) raises -> Tensor:
        """Reinterpret the row-major storage with a new matrix shape."""
        if rows * cols != self.rows * self.cols:
            raise Error("Tensor reshape must preserve the element count")
        return Tensor(self.buffer^, rows, cols)

    def view(
        ref self,
    ) -> TileTensor[BF16, MatrixLayout, origin_of(self.buffer)]:
        return TileTensor(
            self.buffer, row_major(Int32(self.rows), Int32(self.cols))
        )
