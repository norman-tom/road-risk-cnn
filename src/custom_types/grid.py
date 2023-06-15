class Grid:
    def __init__(
        self,
        label: str,
        origin: tuple,
        width: float,
        height: float,
        cell_dim: float,
        res: int = 1,
        pad: bool = False,
    ) -> None:
        """
        :param label: dataset label
        :param origin: easting and northing at the bottom left corner
        :param width: width of grid in meters
        :param height: height of grid in meters
        :param cell_dim: dimension of square grid cell
        :param res: resolution of the grid cells, defaults to 1
        :param pad: defaults to False
        """
        self._label = label
        self._origin = origin
        self._res = res
        self._pad = pad
        self._cell_dim = cell_dim
        self._cells_east = int(width / self._cell_dim)
        self._cells_north = int(height / self._cell_dim)
        self._height = self._cell_dim * self._cells_north
        self._width = self._cell_dim * self._cells_east
        self.__pos: int
        self.__size: int

        if self._cell_dim % res != 0:
            raise Exception("Grid dimension is not a multiple of the resolution")

        if self._pad:
            if (self._cell_dim % 3) == 0:
                self._cells_east = 3 * self._cells_east - 2
                self._cells_north = 3 * self._cells_north - 2
            else:
                raise Exception("(self._cell_dim % 3) != 0")
        self.__size = self._cells_east * self._cells_north

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __iter__(self):
        self.__pos = 0
        return self

    def __next__(self):
        if self.__pos < self.__size:
            tmp = self.__pos
            self.__pos += 1
            return self[tmp]
        else:
            raise StopIteration

    def __getitem__(self, pos):
        if (pos < self.__size) and (pos >= 0):
            row = int(pos / self._cells_east)
            col = pos % self._cells_east
            offset = self._cell_dim / 3 if self._pad else self._cell_dim
            origin = (
                self._origin[0] + col * offset,
                self._origin[1] + row * offset,
            )
            return Grid(
                origin=origin,
                label=f"{self._label}_{pos}",
                res=self._res,
                cell_dim=self._cell_dim,
                width=self._cell_dim,
                height=self._cell_dim,
            )
        else:
            raise IndexError()
        
    def __len__(self):
        return self.__size

    @property
    def origin(self):
        return self._origin

    @property
    def box(self):
        if self._pad:
            return (
                self._origin[0] + (self.cell_dim * self._cells_east + 2 * self._cell_dim) / 3,
                self._origin[1] + (self.cell_dim * self._cells_north + 2 * self._cell_dim) / 3,
            )
        else:
            return (
                self._origin[0] + self.cell_dim * self._cells_east,
                self._origin[1] + self.cell_dim * self._cells_north,
            )

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def cell_dim(self):
        return self._cell_dim

    @property
    def shape(self):
        return (self.__size, self._cells_east, self._cells_north)

    @property
    def res(self):
        return self._res

    @property
    def pixels(self):
        return (int(self._width / self._res), int(self._height / self._res))
