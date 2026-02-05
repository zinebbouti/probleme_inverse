module grille
   use precisions, only: rk
   implicit none
   private

   public :: grid1, grid2

   type :: grid1

      character(:), allocatable :: name

      character(:), allocatable :: scale

      integer :: ncells

      real(rk), allocatable :: edges(:)

      real(rk), allocatable :: center(:)

      real(rk), allocatable :: width(:)

      real(rk), dimension(:), pointer :: left => null()

      real(rk), dimension(:), pointer :: right => null()

   contains
      procedure, pass(self) :: linear => grid1_linear
      procedure, pass(self), private :: clear => grid1_clear
      procedure, pass(self), private :: compute => grid1_compute
   end type grid1

   type :: grid2

      type(grid1) :: x

      type(grid1) :: y

   contains
      procedure, pass(self) :: cartesian => grid2_cartesian
   end type grid2

contains

   pure subroutine grid1_linear(self, xmin, xmax, ncells, name)
 
      class(grid1), intent(inout) :: self
      
      real(rk), intent(in) :: xmin

      real(rk), intent(in) :: xmax

      integer, intent(in) :: ncells

      character(*), intent(in), optional :: name


      real(rk) :: xedges(0:ncells), rx
      integer :: i
      character(:), allocatable :: grid_name

      if (xmax <= xmin) then
         error stop "Invalid input 'xmin', 'xmax'. Valid range: xmax > xmin."
      end if
      if (ncells < 1) then
         error stop "Invalid input 'ncells'. Valid range: ncells > 1."
      end if

      call self%clear

      rx = (xmax - xmin)/ncells
      do concurrent(i=0:ncells)
         xedges(i) = xmin + rx*i
      end do

      if (present(name)) then
         grid_name = name
      else
         grid_name = ""
      end if

      self%scale = "linear"
      call self%compute(xedges, grid_name)

   end subroutine grid1_linear

   pure subroutine grid1_compute(self, xedges, name)
      class(grid1), intent(inout), target :: self

      real(rk), intent(in) :: xedges(0:)

      character(*), intent(in) :: name
        !! grid name

      self%ncells = ubound(xedges, 1)
      self%edges = xedges
      self%left => self%edges(0:self%ncells - 1)
      self%right => self%edges(1:self%ncells)
      self%center = (self%left + self%right)/2
      self%width = self%right - self%left
      self%name = name

   end subroutine grid1_compute

   pure subroutine grid1_clear(self)
   
      class(grid1), intent(inout), target :: self
        

    
      if (allocated(self%name)) deallocate (self%name)
      if (allocated(self%edges)) deallocate (self%edges)
      if (allocated(self%center)) deallocate (self%center)
      if (allocated(self%width)) deallocate (self%width)
      if (associated(self%left)) nullify (self%left)
      if (associated(self%right)) nullify (self%right)
      self%ncells = 0
      self%scale = ""

   end subroutine grid1_clear

   pure subroutine grid2_cartesian(self, xmin, xmax, ymin, ymax, ncells, xname, yname)

      class(grid2), intent(inout) :: self

      real(rk), intent(in) :: xmin

      real(rk), intent(in) :: xmax
      
      real(rk), intent(in) :: ymin
      
      real(rk), intent(in) :: ymax
     
      integer, intent(in) :: ncells(2)
      
      character(*), intent(in), optional :: xname
       
      character(*), intent(in), optional :: yname
       
      character(:), allocatable :: name_x, name_y

    
      if (present(xname)) then
         name_x = xname
      else
         name_x = "x"
      end if

      if (present(yname)) then
         name_y = yname
      else
         name_y = "y"
      end if

     
      call self%x%linear(xmin, xmax, ncells(1), name_x)
      call self%y%linear(ymin, ymax, ncells(2), name_y)

   end subroutine grid2_cartesian

end module grille
