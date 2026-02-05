module schema_weno
   use precisions, only: rk
   implicit none
   private

   public :: weno, c1, c2, c3

   ! Types de conditions aux limites
   integer, parameter, public :: BC_PERIODIC = 1
   integer, parameter, public :: BC_OUTFLOW = 2
   integer, parameter, public :: BC_REFLECTIVE = 3

   real(rk), parameter :: d1(0:0) = 1.0_rk, &
                          d2(0:1) = [2.0_rk/3, 1.0_rk/3], &
                          d3(0:2) = [0.3_rk, 0.6_rk, 0.1_rk]
   real(rk), parameter :: &
      c1(0:0, -1:0) = reshape([1.0_rk, 1.0_rk], [1, 2], order=[1, 2]), &
      c2(0:1, -1:1) = reshape([3.0_rk/2, -1.0_rk/2, 1.0_rk/2, 1.0_rk/2, -1.0_rk/2, 3.0_rk/2], &
                              [2, 3], order=[1, 2]), &
      c3(0:2, -1:2) = reshape([11.0_rk/6, -7.0_rk/6, 1.0_rk/3, 1.0_rk/3, 5.0_rk/6, -1.0_rk/6, &
                               -1.0_rk/6, 5.0_rk/6, 1.0_rk/3, 1.0_rk/3, -7.0_rk/6, 11.0_rk/6], &
                              [3, 4], order=[1, 2])

   type :: weno
      character(:), allocatable :: msg

      integer :: ierr = 0

      integer :: ncells

      integer :: k = 3
         !! ordre de reconstruction (k = 1, 2 or 3)
      integer :: bc_type = BC_PERIODIC

      real(rk) :: eps = 1e-6_rk

      real(rk), allocatable, private :: d(:)

      real(rk), allocatable, private :: c(:, :)

   contains
      procedure, pass(self) :: reconstruct => weno_reconstruct
      procedure, pass(self) :: set_bc => weno_set_bc
   end type weno

   interface weno
      module procedure :: weno_init
   end interface weno

contains

   pure type(weno) function weno_init(ncells, k, eps, bc_type) result(self)

      integer, intent(in) :: ncells

      integer, intent(in), optional :: k
         !! (2k - 1) est l'ordre de reconstruction  (1 <= k <= 3).
         !! k = 3 est le 5ème ordre 
      real(rk), intent(in), optional :: eps

      integer, intent(in), optional :: bc_type

      if (ncells > 0) then
         self%ncells = ncells
      else
         self%msg = "Invalid input 'ncells'. Valid range: ncells > 0."
         self%ierr = 1
         error stop self%msg
      end if

      if (present(k)) then
         if ((k >= 1) .and. (k <= 3)) then
            self%k = k
         else
            self%msg = "Invalid input 'k'. Valid range: 1 <= k <= 3."
            self%ierr = 1
            error stop self%msg
         end if
      end if

      if (present(eps)) then
         if (eps > epsilon(1.0_rk)) then
            self%eps = eps
         else
            self%msg = "Invalid input 'eps'. Valid range: eps > epsilon."
            self%ierr = 1
            error stop self%msg
         end if
      end if

      if (present(bc_type)) then
   if ( (bc_type == BC_PERIODIC)  .or. &
        (bc_type == BC_OUTFLOW)   .or. &
        (bc_type == BC_REFLECTIVE) ) then

      self%bc_type = bc_type

   else
      self%msg = "Invalid input 'bc_type'. Valid values: BC_PERIODIC, BC_OUTFLOW or BC_REFLECTIVE."
      self%ierr = 1
      error stop self%msg
   end if
end if


      select case (self%k)
      case (1)
         self%d = d1
         self%c = c1
      case (2)
         self%d = d2
         self%c = c2
      case (3)
         self%d = d3
         self%c = c3
      end select

   end function weno_init

   pure subroutine weno_set_bc(self, bc_type)

      class(weno), intent(inout) :: self
      integer, intent(in) :: bc_type
      self%bc_type = bc_type
   end subroutine weno_set_bc

   pure subroutine weno_reconstruct(self, v, vl, vr)
   !!   Cette subroutine implémente la méthode WENO d'ordre (2k-1) pour des schémas 
   !! volumes finis sur grilles uniformes .

   
      class(weno), intent(in) :: self

      real(rk), intent(in) :: v(:)

      real(rk), intent(out) :: vl(:)

      real(rk), intent(out) :: vr(:)


      real(rk), dimension(0:self%k - 1) :: vlr, vrr, w, wtilde, alfa, alfatilde, beta
      real(rk), dimension(1 - (self%k - 1):self%ncells + (self%k - 1)) :: vext
      integer :: i, r

      
      associate (k => self%k, nc => self%ncells, eps => self%eps, d => self%d, c => self%c)
         
         vext(1:nc) = v
         
         select case (self%bc_type)
         
         case (BC_PERIODIC)
            ! Extension périodique
            vext(:0) = v(nc-k+2:nc)        ! Cellules de droite
            vext(nc+1:) = v(1:k-1)         ! Cellules de gauche
            
         case (BC_OUTFLOW)
            ! Extrapolation d'ordre 0 (constant)
            vext(:0) = v(1)
            vext(nc+1:) = v(nc)
            case (BC_REFLECTIVE)

   ! Côté gauche
   vext(0)  = v(1)
   vext(-1) = v(2)

   ! Côté droit
   vext(nc+1) = v(nc)
   vext(nc+2) = v(nc-1)

            
         end select
         
         do concurrent(i=1:nc)

            do concurrent(r=0:k - 1)
               vrr(r) = sum(c(:, r)*vext(i - r:i - r + k - 1))
               vlr(r) = sum(c(:, r - 1)*vext(i - r:i - r + k - 1))
            end do

            select case (k)

            case (1)
               beta(0) = 0.0_rk

            case (2)
               beta(0) = (vext(i + 1) - vext(i))**2
               beta(1) = (vext(i) - vext(i - 1))**2

            case (3)
               beta(0) = 13.0_rk/12*(vext(i) - 2*vext(i + 1) + vext(i + 2))**2 &
                         + 1.0_rk/4*(3*vext(i) - 4*vext(i + 1) + vext(i + 2))**2

               beta(1) = 13.0_rk/12*(vext(i - 1) - 2*vext(i) + vext(i + 1))**2 &
                         + 1.0_rk/4*(vext(i - 1) - vext(i + 1))**2

               beta(2) = 13.0_rk/12*(vext(i - 2) - 2*vext(i - 1) + vext(i))**2 &
                         + 1.0_rk/4*(vext(i - 2) - 4*vext(i - 1) + 3*vext(i))**2

            end select

            alfa = d/(eps + beta)**2
            alfatilde = d(k - 1:0:-1)/(eps + beta)**2
            w = alfa/sum(alfa)
            wtilde = alfatilde/sum(alfatilde)

            vr(i) = sum(w*vrr)
            vl(i) = sum(wtilde*vlr)

         end do
      end associate

   end subroutine weno_reconstruct

end module schema_weno