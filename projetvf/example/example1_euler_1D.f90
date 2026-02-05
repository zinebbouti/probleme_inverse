program example1_euler_1d_sod
   use, intrinsic :: iso_fortran_env, only: stderr => error_unit, stdout => output_unit
   use precisions, only: rk
   use schema_temps, only: rktvd
       use schema_weno, only: weno, BC_PERIODIC, BC_OUTFLOW
 
   use grille, only: grid1
   implicit none

   integer, parameter :: nc = 100
   integer, parameter :: neq = 3
   real(rk) :: u(neq*nc), dt, time, time_out, time_start, time_end
   integer :: num_time_points, i, j, ii
   type(grid1) :: gx
   type(weno) :: myweno
   type(rktvd) :: ode
   real(rk), parameter :: gamma = 1.4_rk

   call gx%linear(xmin=-5.0_rk, xmax=5.0_rk, ncells=nc)

   myweno = weno(ncells=nc, k=2, eps=1e-6_rk,bc_type=BC_OUTFLOW)

   call output(1)

   do i = 1, nc
       do j = 1, neq
           u((j-1)*nc + i) = ic(gx%center(i), j)
       end do
   end do

   ode = rktvd(rhs, neq*nc, order=3)

   time_start = 0.0_rk
   time_end   = 1.3_rk  
   dt         = 1e-3_rk 

   time = time_start
   num_time_points = 100
   do ii = 0, num_time_points
      time_out = time_end*ii/num_time_points
      call ode%integrate(u, time, time_out, dt)
      call output(2)
   end do

   call output(3)

contains


   pure subroutine rhs(t, v, vdot)
      real(rk), intent(in) :: t
      real(rk), intent(in) :: v(:)
      real(rk), intent(out) :: vdot(:)
      real(rk) :: fedges(neq,0:nc)
      real(rk) :: vl(neq,nc), vr(neq,nc)
      real(rk) :: vl_local(nc), vr_local(nc)
      real(rk) :: ul(neq), ur(neq), fnum(neq)
      integer :: i, j

      do j = 1, neq
          call myweno%reconstruct(v((j-1)*nc+1:j*nc), vl_local, vr_local)
          vl(j,:) = vl_local
          vr(j,:) = vr_local
      end do

      do i = 1, nc-1
          ur = vr(:,i)     ! Right state of left cell
          ul = vl(:,i+1)   ! Left state of right cell
          fnum = flux_lax_friedrichs(ur, ul)
          fedges(:,i) = fnum
      end do

      fedges(:,0)  = fedges(:,1)
      fedges(:,nc) = fedges(:,nc-1)

      ! Compute du/dt = -(f_{i+1/2} - f_{i-1/2})/dx
      do j = 1, neq
          do i = 1, nc
              vdot((j-1)*nc + i) = -(fedges(j,i) - fedges(j,i-1))/gx%width(i)
          end do
      end do

   end subroutine rhs

   pure function flux_lax_friedrichs(ul, ur) result(fnum)
      real(rk), intent(in) :: ul(neq), ur(neq)
      real(rk) :: fnum(neq)
      real(rk) :: fl(neq), fr(neq), alpha
      real(rk) :: rho_l, u_l, p_l, a_l
      real(rk) :: rho_r, u_r, p_r, a_r

      ! Gauche
      fl = flux_euler_1d(ul)
      rho_l = ul(1)
      u_l = ul(2)/rho_l
      p_l = (gamma-1.0_rk)*(ul(3) - 0.5_rk*rho_l*u_l**2)
      a_l = sqrt(gamma*p_l/rho_l)

      ! Droite
      fr = flux_euler_1d(ur)
      rho_r = ur(1)
      u_r = ur(2)/rho_r
      p_r = (gamma-1.0_rk)*(ur(3) - 0.5_rk*rho_r*u_r**2)
      a_r = sqrt(gamma*p_r/rho_r)

     
      alpha = max(abs(u_l) + a_l, abs(u_r) + a_r)

      ! Lax-Friedrichs flux
      fnum = 0.5_rk*(fl + fr) - 0.5_rk*alpha*(ur - ul)

   end function flux_lax_friedrichs

   pure function flux_euler_1d(v) result(f)
      real(rk), intent(in)  :: v(neq)
      real(rk) :: f(neq)
      real(rk) :: rho, uvel, E, p

      rho  = v(1)
      uvel = v(2)/rho
      E    = v(3)
      p    = (gamma-1.0_rk)*(E - 0.5_rk*rho*uvel**2)

      f(1) = rho*uvel
      f(2) = rho*uvel**2 + p
      f(3) = uvel*(E+p)
   end function flux_euler_1d

   elemental real(rk) function ic(x, var)
      real(rk), intent(in) :: x
      integer, intent(in) :: var
      real(rk) :: rho, uvel, p

      ! Sod shock tube problem
      if (x < 0.0_rk) then
          rho = 1.0_rk
          uvel = 0.0_rk
          p = 1.0_rk
      else
          rho = 0.125_rk
          uvel = 0.0_rk
          p = 0.1_rk
      end if

      select case(var)
      case(1)
          ic = rho
      case(2)
          ic = rho*uvel
      case(3)
          ic = p/(gamma-1.0_rk) + 0.5_rk*rho*uvel**2
      end select
   end function ic

   subroutine output(action)
      integer, intent(in) :: action
      character(*), parameter :: folder = "./resultats/example1/"
      real(rk), save :: cpu_start=0.0_rk, cpu_end=0.0_rk
      integer, save :: funit_x=0, funit_u=0
      integer :: i
      real(rk) :: rho, mom, E, uvel, p

      select case(action)
      case(1)
         write (stdout, '(1x,a)') "Running Euler 1D example ..."
         call cpu_time(cpu_start)
         open(newunit=funit_x, file=folder//"x.txt", status="replace")
         write(funit_x,'(a5,2(1x,a15))') "i","x(i)","dx(i)"
         do i=1,nc
            write(funit_x,'(i5,2(1x,es15.5))') i, gx%center(i), gx%width(i)
         end do

         open(newunit=funit_u, file=folder//"u.txt", status="replace")
         write(funit_u,'(a16,4(1x,a15))') "t","x","rho","u","p"
      case(2)
         do i=1,nc
            rho = u((0)*nc + i)
            mom = u((1)*nc + i)
            E   = u((2)*nc + i)
            uvel = mom/rho
            p = (gamma-1.0_rk)*(E - 0.5_rk*rho*uvel**2)
            write(funit_u,'(es16.5e3,4(1x,es15.5))') time, gx%center(i), rho, uvel, p
         end do
         write(funit_u,*)
      case(3)
         close(funit_x)
         close(funit_u)
         call cpu_time(cpu_end)
         write(stdout,'(1x,a,1x,f6.1)') "Elapsed time (ms) :", 1e3_rk*(cpu_end - cpu_start)
      end select
   end subroutine output

end program example1_euler_1d_sod
