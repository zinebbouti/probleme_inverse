program euler_2d
   use, intrinsic :: iso_fortran_env, only: stderr => error_unit, stdout => output_unit
   use precisions, only: rk
   use schema_temps, only: rktvd
    use schema_weno, only: weno, BC_PERIODIC, BC_OUTFLOW
    use grille, only: grid2
   implicit none

   integer, parameter :: neq = 4
   integer, parameter :: nc(2) = [160,160]
   real(rk) :: u(neq*product(nc)), dt, time, time_out, time_start, time_end
   integer :: ii,jj,num_time_points
   type(grid2) :: gx2d
   type(weno) :: myweno(2)
   type(rktvd) :: ode
   real(rk), parameter :: gamma = 1.4_rk

   !-----------------------------
   ! Grille 2D cartésienne
   !-----------------------------
   call gx2d%cartesian(xmin=0.0_rk, xmax=10.0_rk, ymin=0.0_rk, ymax=10.0_rk, ncells=nc)

   !-----------------------------
   ! Initialisation WENO
   !-----------------------------
   myweno(1) = weno(ncells=nc(1), k=3, eps=1e-10_rk,bc_type=BC_PERIODIC)
   myweno(2) = weno(ncells=nc(2), k=3, eps=1e-10_rk,bc_type=BC_PERIODIC)

   !-----------------------------
   ! Conditions initiales
   !-----------------------------
   do concurrent(ii=1:nc(1), jj=1:nc(2))
       call ic([gx2d%x%center(ii), gx2d%y%center(jj)], &
               u((jj-1)*nc(1)*neq + (ii-1)*neq + 1))
   end do

   !-----------------------------
   ! ODE Solver
   !-----------------------------
   ode = rktvd(rhs, size(u), 3)  ! RK3-TVD
   time_start = 0.0_rk
   time_end   =50.0_rk   ! Commencer avec t=1 pour tester

   
   !dt=2e-2_rk !pour 40*40
   !dt=1.0e-2_rk ! pour 80
   dt=  5e-3_rk !pour 160
   num_time_points = 50
   time = time_start

   do ii = 0,num_time_points
       time_out = time_end*ii/num_time_points
       call ode%integrate(u, time, time_out, dt)
       call output2d(u, time)
   end do

contains

   !========================
   subroutine rhs(t, v, vdot)
       real(rk), intent(in) :: t
       real(rk), intent(in) :: v(:)
       real(rk), intent(out):: vdot(:)
       real(rk) :: vl(neq,nc(1)), vr(neq,nc(1))
       real(rk) :: vl_local(nc(1)), vr_local(nc(1))
       real(rk) :: vtemp(nc(1))
       real(rk) :: fedges_x(neq,0:nc(1),1:nc(2))
       real(rk) :: fedges_y(neq,0:nc(2),1:nc(1))
       real(rk) :: ul(neq), ur(neq), fnum(neq)
       integer :: i,j,k

       !-----------------------------
       ! Flux x-direction
       !-----------------------------
       do j = 1,nc(2)
           do k = 1,neq
               ! Extraire la k-ième composante le long de x
               do i = 1,nc(1)
                   vtemp(i) = v((j-1)*nc(1)*neq + (i-1)*neq + k)
               end do
               call myweno(1)%reconstruct(vtemp, vl_local, vr_local)
               vl(k,:) = vl_local
               vr(k,:) = vr_local
           end do
           
           ! Flux internes
           do i = 1,nc(1)-1
               ul = vr(:,i)
               ur = vl(:,i+1)
               fnum = flux_lax_friedrichs_2d(ul, ur, gamma, 1)
               fedges_x(:,i,j) = fnum
           end do
           
           ! Conditions aux limites périodiques
           ul = vr(:,nc(1))
           ur = vl(:,1)
           fedges_x(:,0,j) = flux_lax_friedrichs_2d(ul, ur, gamma, 1)
           fedges_x(:,nc(1),j) = fedges_x(:,0,j)
       end do

       !-----------------------------
       ! Flux y-direction
       !-----------------------------
       do i = 1,nc(1)
           do k = 1,neq
               ! Extraire la k-ième composante le long de y
               do j = 1,nc(2)
                   vtemp(j) = v((j-1)*nc(1)*neq + (i-1)*neq + k)
               end do
               call myweno(2)%reconstruct(vtemp, vl_local, vr_local)
               vl(k,:) = vl_local
               vr(k,:) = vr_local
           end do
           
           ! Flux internes
           do j = 1,nc(2)-1
               ul = vr(:,j)
               ur = vl(:,j+1)
               fnum = flux_lax_friedrichs_2d(ul, ur, gamma, 2)
               fedges_y(:,j,i) = fnum
           end do
           
           ! Conditions aux limites périodiques
           ul = vr(:,nc(2))
           ur = vl(:,1)
           fedges_y(:,0,i) = flux_lax_friedrichs_2d(ul, ur, gamma, 2)
           fedges_y(:,nc(2),i) = fedges_y(:,0,i)
       end do

       !-----------------------------
       ! du/dt (divergence des flux)
       !-----------------------------
       do j = 1,nc(2)
           do i = 1,nc(1)
               do k = 1,neq
                   vdot((j-1)*nc(1)*neq + (i-1)*neq + k) = &
                        - (fedges_x(k,i,j) - fedges_x(k,i-1,j))/gx2d%x%width(i) &
                        - (fedges_y(k,j,i) - fedges_y(k,j-1,i))/gx2d%y%width(j)
               end do
           end do
       end do
   end subroutine rhs

   !========================
   pure function flux_lax_friedrichs_2d(ul, ur, gamma, dir) result(fnum)
       real(rk), intent(in) :: ul(neq), ur(neq)
       real(rk), intent(in) :: gamma
       integer, intent(in) :: dir
       real(rk) :: fnum(neq)
       real(rk) :: fl(neq), fr(neq), alpha
       real(rk) :: rho_l, u_l, v_l, p_l, a_l
       real(rk) :: rho_r, u_r, v_r, p_r, a_r

       fl = flux_euler_2d(ul, gamma, dir)
       fr = flux_euler_2d(ur, gamma, dir)

       ! Left state
       rho_l = ul(1)
       u_l = ul(2)/rho_l
       v_l = ul(3)/rho_l
       p_l = (gamma-1.0_rk)*(ul(4) - 0.5_rk*rho_l*(u_l**2 + v_l**2))
       a_l = sqrt(gamma*p_l/rho_l)

       ! Right state
       rho_r = ur(1)
       u_r = ur(2)/rho_r
       v_r = ur(3)/rho_r
       p_r = (gamma-1.0_rk)*(ur(4) - 0.5_rk*rho_r*(u_r**2 + v_r**2))
       a_r = sqrt(gamma*p_r/rho_r)

       ! Max wave speed
       if (dir==1) then
           alpha = max(abs(u_l)+a_l, abs(u_r)+a_r)
       else
           alpha = max(abs(v_l)+a_l, abs(v_r)+a_r)
       end if

       fnum = 0.5_rk*(fl + fr) - 0.5_rk*alpha*(ur - ul)
   end function flux_lax_friedrichs_2d

   !========================
   pure function flux_euler_2d(v, gamma, dir) result(f)
       real(rk), intent(in) :: v(neq)
       real(rk), intent(in) :: gamma
       integer, intent(in) :: dir
       real(rk) :: f(neq)
       real(rk) :: rho, u, vvel, E, p

       rho = v(1)
       u = v(2)/rho
       vvel = v(3)/rho
       E = v(4)
       p = (gamma-1.0_rk)*(E - 0.5_rk*rho*(u**2 + vvel**2))

       if (dir==1) then
           f(1) = rho*u
           f(2) = rho*u**2 + p
           f(3) = rho*u*vvel
           f(4) = (E+p)*u
       else
           f(1) = rho*vvel
           f(2) = rho*u*vvel
           f(3) = rho*vvel**2 + p
           f(4) = (E+p)*vvel
       end if
   end function flux_euler_2d

   !========================
   pure subroutine ic(x, uvec)
       real(rk), intent(in) :: x(2)
       real(rk), intent(out) :: uvec(neq)
       real(rk), parameter :: gamma = 1.4_rk
       real(rk), parameter :: rho_inf = 1.0_rk, u_inf = 1.0_rk, v_inf = 1.0_rk
       real(rk), parameter :: epsilon = 5.0_rk
       real(rk) :: xc, yc, r2, du, dv, dT
       real(rk) :: rho, uvel, vvel, T, p, E
       real(rk), parameter :: pi = acos(-1.0_rk)

       ! Centre du vortex
       xc = x(1) - 5.0_rk
       yc = x(2) - 5.0_rk
       r2 = xc*xc + yc*yc

       ! Perturbations du vortex (isentropique)
       du = - epsilon/(2.0_rk*pi) * yc * exp(0.5_rk*(1.0_rk - r2))
       dv =   epsilon/(2.0_rk*pi) * xc * exp(0.5_rk*(1.0_rk - r2))
       dT = - (gamma-1.0_rk) * epsilon**2 / (8.0_rk*gamma*pi**2) * exp(1.0_rk - r2)

       ! Variables primitives avec relation isentropique
       ! Pour un gaz parfait isentropique: T = rho^(gamma-1)
       T = 1.0_rk + dT
       rho = T**(1.0_rk/(gamma-1.0_rk))
       p = rho**gamma  ! Relation isentropique p = rho^gamma
       uvel = u_inf + du
       vvel = v_inf + dv

       ! Variables conservées
       E = p/(gamma-1.0_rk) + 0.5_rk*rho*(uvel**2 + vvel**2)
       uvec = [rho, rho*uvel, rho*vvel, E]
   end subroutine ic

   !========================
   subroutine output2d(u, time)
       use, intrinsic :: iso_fortran_env, only: output_unit
       implicit none
       real(rk), intent(in) :: u(:)
       real(rk), intent(in) :: time
       character(*), parameter :: folder = "./resultats/example2/"
       integer, save :: funit_x = 0, funit_u = 0
       integer :: i, j, k
       real(rk) :: rho, uvel, vvel, E, p

       !-----------------------------
       ! Ouverture des fichiers une seule fois
       !-----------------------------
       if (funit_x == 0) then
           open(newunit=funit_x, file=folder//"x.txt", status="replace")
           write(funit_x,'(a)') "i,j,x(i),y(j)"
           do j = 1, nc(2)
               do i = 1, nc(1)
                   write(funit_x,'(i5,i5,2(1x,es15.5))') i, j, gx2d%x%center(i), gx2d%y%center(j)
               end do
           end do
       end if

       if (funit_u == 0) then
           open(newunit=funit_u, file=folder//"u.txt", status="replace")
           write(funit_u,'(a)') "time,i,j,rho,u,v,p"
       end if

       !-----------------------------
       ! Écriture des snapshots (indexation cohérente avec rhs)
       !-----------------------------
       do j = 1, nc(2)
           do i = 1, nc(1)
               rho  = u((j-1)*nc(1)*neq + (i-1)*neq + 1)
               uvel = u((j-1)*nc(1)*neq + (i-1)*neq + 2)/rho
               vvel = u((j-1)*nc(1)*neq + (i-1)*neq + 3)/rho
               E    = u((j-1)*nc(1)*neq + (i-1)*neq + 4)
               p    = (gamma-1.0_rk)*(E - 0.5_rk*rho*(uvel**2 + vvel**2))
               write(funit_u,'(es16.5e3,2(i5,1x),4(es15.5))') time, i, j, rho, uvel, vvel, p
           end do
       end do
       write(funit_u,*)  

   end subroutine output2d

end program euler_2d
