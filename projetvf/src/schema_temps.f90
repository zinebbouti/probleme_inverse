module schema_temps

   use precisions, only: rk
   implicit none
   private

   public :: rktvd

   type :: rktvd
      procedure(integrand), pointer, nopass, private :: fu => null()

      integer :: neq

      integer :: order

      integer :: fevals = 0

      integer :: istate = 0

      character(:), allocatable :: msg

      real(rk), allocatable, private :: ui(:)
      real(rk), allocatable, private :: udot(:)
      real(rk), allocatable, private :: k1(:), k2(:), k3(:), k4(:)  ! Pour RK4
   contains
      procedure, pass(self) :: integrate => rktvd_integrate
      procedure, pass(self), private :: error_msg
   end type rktvd

   abstract interface
      subroutine integrand(t, u, udot)
         import :: rk
         real(rk), intent(in) :: t, u(:)
         real(rk), intent(out) :: udot(:)
      end subroutine
   end interface

   interface rktvd
      module procedure :: rktvd_init
   end interface rktvd

contains

   type(rktvd) function rktvd_init(fu, neq, order) result(self)

      procedure(integrand) :: fu
      integer, intent(in) :: neq
      integer, intent(in) :: order

      self%fu => fu

      if (neq > 0) then
         self%neq = neq
      else
         call self%error_msg("Invalid input 'neq'. Valid range: neq >= 1.")
      end if

      if ((order >= 1) .and. (order <= 4)) then
         self%order = order
      else
         call self%error_msg("Invalid input 'order' in 'rktvd'. Valid range: 1 <= k <= 4.")
      end if

      allocate(self%ui(self%neq), self%udot(self%neq))
      
      ! Allocation pour RK4
      if (order == 4) then
         allocate(self%k1(self%neq), self%k2(self%neq), self%k3(self%neq), self%k4(self%neq))
      end if
      
      self%istate = 1

   end function rktvd_init

   subroutine rktvd_integrate(self, u, t, tout, dt, itask)

      class(rktvd), intent(inout) :: self
      real(rk), intent(inout) :: u(:)
      real(rk), intent(inout) :: t
      real(rk), intent(in) :: tout
      real(rk), intent(in) :: dt
      integer, intent(in), optional :: itask

      integer :: itask_

      if (self%istate < 1) return
      if (is_done(t, tout, dt)) return
      
      if (present(itask)) then
         itask_ = itask
      else
         itask_ = 1
      end if

      associate (ui => self%ui, udot => self%udot)
         select case (self%order)

            ! ----------------------- Euler explicite (ordre 1) -----------------------
         case (1)
            do
               call self%fu(t, u, udot)
               u = u + dt*udot
               t = t + dt
               self%fevals = self%fevals + 1
               if (is_done(t, tout, dt) .or. itask_ == 2) exit
            end do

            ! ----------------------- RK2-TVD (ordre 2) -----------------------
            ! Équation (4.10), page 43
         case (2)
            do
               call self%fu(t, u, udot)
               ui = u + dt*udot
               call self%fu(t + dt, ui, udot)
               u = (u + ui + dt*udot)/2
               t = t + dt
               self%fevals = self%fevals + 2
               if (is_done(t, tout, dt) .or. itask_ == 2) exit
            end do

            ! ----------------------- RK3-TVD (ordre 3) - OPTIMAL -----------------------
            ! Équation (4.11), page 43
         case (3)
            do
               call self%fu(t, u, udot)
               ui = u + dt*udot
               call self%fu(t + dt, ui, udot)
               ui = (3*u + ui + dt*udot)/4
               call self%fu(t + dt/2, ui, udot)
               u = (u + 2*ui + 2*dt*udot)/3
               t = t + dt
               self%fevals = self%fevals + 3
               if (is_done(t, tout, dt) .or. itask_ == 2) exit
            end do

            ! ----------------------- RK4 classique (ordre 4) -----------------------
            ! Méthode de Runge-Kutta classique d'ordre 4
         case (4)
            associate (k1 => self%k1, k2 => self%k2, k3 => self%k3, k4 => self%k4)
               do
                  ! k1 = f(t, u)
                  call self%fu(t, u, k1)
                  
                  ! k2 = f(t + dt/2, u + dt*k1/2)
                  ui = u + 0.5_rk*dt*k1
                  call self%fu(t + 0.5_rk*dt, ui, k2)
                  
                  ! k3 = f(t + dt/2, u + dt*k2/2)
                  ui = u + 0.5_rk*dt*k2
                  call self%fu(t + 0.5_rk*dt, ui, k3)
                  
                  ! k4 = f(t + dt, u + dt*k3)
                  ui = u + dt*k3
                  call self%fu(t + dt, ui, k4)
                  
                  ! u_n+1 = u_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
                  u = u + (dt/6.0_rk)*(k1 + 2.0_rk*k2 + 2.0_rk*k3 + k4)
                  
                  t = t + dt
                  self%fevals = self%fevals + 4
                  if (is_done(t, tout, dt) .or. itask_ == 2) exit
               end do
            end associate

         end select
      end associate

      if (self%istate == 1) self%istate = 2

   end subroutine rktvd_integrate

   pure logical function is_done(t, tout, dt)
      real(rk), intent(in) :: t, tout, dt
      is_done = (t - tout)*sign(1.0_rk, dt) > 0.0_rk
   end function is_done

   pure subroutine error_msg(self, msg)
      class(rktvd), intent(inout) :: self
      character(*), intent(in) :: msg
      self%msg = msg
      self%istate = -1
      error stop self%msg
   end subroutine error_msg

end module schema_temps