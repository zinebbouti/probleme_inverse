program double_mach
  use, intrinsic :: iso_fortran_env, only: stderr => error_unit, stdout => output_unit
  use precisions,  only: rk
  use schema_temps,only: rktvd
  use schema_weno, only: weno, BC_OUTFLOW
  use grille,      only: grid2
  implicit none

  integer, parameter :: neq = 4
  integer, parameter :: nc(2) = [480,119]
  real(rk), parameter :: gamma = 1.4_rk
  real(rk), parameter :: CFL   = 0.6_rk

  type(grid2) :: g
  type(weno)  :: myweno(2)
  type(rktvd) :: ode

  real(rk), allocatable :: U(:)
  real(rk) :: dt, time, time_end

  !-----------------------------
  ! Grille 2D cartésienne
  !-----------------------------
  call g%cartesian(xmin=0.0_rk, xmax=4.0_rk, ymin=0.0_rk, ymax=1.0_rk, ncells=nc)
  allocate(U(neq*product(nc)))

  !-----------------------------
  ! Initialisation WENO (5ème ordre si k=3)
  !-----------------------------
  myweno(1) = weno(ncells=nc(1), k=3, eps=1e-6_rk, bc_type=BC_OUTFLOW)
  myweno(2) = weno(ncells=nc(2), k=3, eps=1e-6_rk, bc_type=BC_OUTFLOW)

  !-----------------------------
  ! Conditions initiales (t=0)
  !-----------------------------
  call init_double_mach(U)

  !-----------------------------
  ! ODE Solver
  !-----------------------------
  ode      = rktvd(rhs, size(U), 3)     ! RK3-TVD
  time     = 0.0_rk
  time_end = 0.2_rk

  dt = compute_dt_cfl(U)
  call ode%integrate(U, time, time_end, dt)

  call output2d(U, time_end)

contains

  !=========================================================
  integer function idx(i,j,k)
    integer, intent(in) :: i,j,k
    idx = (j-1)*nc(1)*neq + (i-1)*neq + k
  end function idx

  !=========================================================
  subroutine get_cell(v, i, j, Uc)
    real(rk), intent(in)  :: v(:)
    integer, intent(in)   :: i, j
    real(rk), intent(out) :: Uc(neq)
    integer :: k
    do k=1,neq
      Uc(k) = v(idx(i,j,k))
    end do
  end subroutine get_cell

  !=========================================================
  subroutine set_cell(v, i, j, Uc)
    real(rk), intent(inout) :: v(:)
    integer, intent(in)     :: i, j
    real(rk), intent(in)    :: Uc(neq)
    integer :: k
    do k=1,neq
      v(idx(i,j,k)) = Uc(k)
    end do
  end subroutine set_cell

  !=========================================================
  subroutine rhs(t, v, vdot)
    real(rk), intent(in)  :: t
    real(rk), intent(in)  :: v(:)
    real(rk), intent(out) :: vdot(:)

    integer :: i,j,k
    real(rk) :: UL(neq), UR(neq)

    ! Buffers WENO
    real(rk) :: vlx(neq,nc(1)), vrx(neq,nc(1))
    real(rk) :: vly(neq,nc(2)), vry(neq,nc(2))
    real(rk) :: vtemp(max(nc(1),nc(2)))
    real(rk) :: vl_local(max(nc(1),nc(2))), vr_local(max(nc(1),nc(2)))

    ! flux aux interfaces
    real(rk), allocatable, save :: fedges_x(:,:,:)
    real(rk), allocatable, save :: fedges_y(:,:,:)

    real(rk) :: xedge, yedge

    if (.not. allocated(fedges_x)) allocate(fedges_x(neq,0:nc(1),1:nc(2)))
    if (.not. allocated(fedges_y)) allocate(fedges_y(neq,0:nc(2),1:nc(1)))

    vdot = 0.0_rk

    !------------------------------------------------------
    ! Flux en x (pour chaque ligne j)
    !------------------------------------------------------
    do j = 1,nc(2)

      ! reconstruction WENO sur la ligne j
      do k = 1,neq
        do i = 1,nc(1)
          vtemp(i) = v(idx(i,j,k))
        end do
        call myweno(1)%reconstruct(vtemp(1:nc(1)), vl_local(1:nc(1)), vr_local(1:nc(1)))
        vlx(k,:) = vl_local(1:nc(1))
        vrx(k,:) = vr_local(1:nc(1))
      end do

      ! ---- interface gauche i=0 : inflow choc mobile (exact motion)
      yedge = g%y%center(j)
      xedge = g%x%center(1) - 0.5_rk*g%x%width(1)
      call shock_state(xedge, yedge, t, UL)
      UR = vlx(:,1)
      fedges_x(:,0,j) = flux_lf(UL, UR, 1)

      ! ---- interfaces internes
      do i = 1,nc(1)-1
        UL = vrx(:,i)
        UR = vlx(:,i+1)
        fedges_x(:,i,j) = flux_lf(UL, UR, 1)
      end do

      ! ---- interface droite i=nc(1) : outflow (copie)
      UL = vrx(:,nc(1))
      UR = vrx(:,nc(1))
      fedges_x(:,nc(1),j) = flux_lf(UL, UR, 1)

    end do

    !------------------------------------------------------
    ! Flux en y (pour chaque colonne i)
    !------------------------------------------------------
    do i = 1,nc(1)

      ! reconstruction WENO sur la colonne i
      do k = 1,neq
        do j = 1,nc(2)
          vtemp(j) = v(idx(i,j,k))
        end do
        call myweno(2)%reconstruct(vtemp(1:nc(2)), vl_local(1:nc(2)), vr_local(1:nc(2)))
        vly(k,:) = vl_local(1:nc(2))
        vry(k,:) = vr_local(1:nc(2))
      end do

      xedge = g%x%center(i)

      ! ---- bas j=0 : BC du papier
      !     x in [0,1/6] => post-choc exact (PAS de t)
      !     x > 1/6      => mur slip (vy -> -vy)
      UR = vly(:,1)                    ! état intérieur
      call bottom_bc(xedge, UR, UL)    ! UL = ghost bas
      fedges_y(:,0,i) = flux_lf(UL, UR, 2)

      ! ---- interfaces internes
      do j = 1,nc(2)-1
        UL = vry(:,j)
        UR = vly(:,j+1)
        fedges_y(:,j,i) = flux_lf(UL, UR, 2)
      end do

      ! ---- haut j=nc(2) : inflow choc mobile (exact motion)
      yedge = g%y%center(nc(2)) + 0.5_rk*g%y%width(nc(2))
      call shock_state(xedge, yedge, t, UR)   ! ghost haut
      UL = vry(:,nc(2))
      fedges_y(:,nc(2),i) = flux_lf(UL, UR, 2)

    end do

    !------------------------------------------------------
    ! vdot = -div(F)
    !------------------------------------------------------
    do j = 1,nc(2)
      do i = 1,nc(1)
        do k = 1,neq
          vdot(idx(i,j,k)) = &
            - (fedges_x(k,i,j) - fedges_x(k,i-1,j))/g%x%width(i) &
            - (fedges_y(k,j,i) - fedges_y(k,j-1,i))/g%y%width(j)
        end do
      end do
    end do

  end subroutine rhs

  !=========================================================
  subroutine shock_state(x, y, t, Uc)
    ! Mach 10 shock, 60°, moving right at speed 10:
    ! x_s(y,t) = 1/6 + y/sqrt(3) + 10 t
    real(rk), intent(in)  :: x, y, t
    real(rk), intent(out) :: Uc(neq)
    real(rk) :: rho, uvel, vvel, p
    real(rk), parameter :: invsqrt3 = 1.0_rk/sqrt(3.0_rk)

    if (x < (1.0_rk/6.0_rk + y*invsqrt3 + 10.0_rk*t)) then
      rho  = 8.0_rk
      uvel = 8.25_rk*sqrt(3.0_rk)/2.0_rk
      vvel = -4.125_rk
      p    = 116.5_rk
    else
      rho  = 1.4_rk
      uvel = 0.0_rk
      vvel = 0.0_rk
      p    = 1.0_rk
    end if

    Uc = prim_to_cons(rho, uvel, vvel, p)
  end subroutine shock_state

  !=========================================================
  subroutine bottom_bc(x, Ucell, Ughost)
    ! Bottom boundary from paper:
    !   0 <= x <= 1/6  : impose exact post-shock (constant)
    !   x >  1/6       : reflective slip wall (vy -> -vy)
    real(rk), intent(in)  :: x
    real(rk), intent(in)  :: Ucell(neq)
    real(rk), intent(out) :: Ughost(neq)
    real(rk) :: rho, uvel, vvel, p

    if (x <= (1.0_rk/6.0_rk)) then
      rho  = 8.0_rk
      uvel = 8.25_rk*sqrt(3.0_rk)/2.0_rk
      vvel = -4.125_rk
      p    = 116.5_rk
      Ughost = prim_to_cons(rho, uvel, vvel, p)
    else
      call cons_to_prim(Ucell, rho, uvel, vvel, p)
      Ughost = prim_to_cons(rho, uvel, -vvel, p)
    end if
  end subroutine bottom_bc

  !=========================================================
  function compute_dt_cfl(v) result(dtloc)
    real(rk), intent(in) :: v(:)
    real(rk) :: dtloc
    real(rk) :: rho,uvel,vvel,p,a, smax, dxmin, dymin
    real(rk) :: Uc(neq)
    integer :: i,j

    smax = 1.0e-12_rk
    do j=1,nc(2)
      do i=1,nc(1)
        call get_cell(v,i,j,Uc)
        call cons_to_prim(Uc, rho, uvel, vvel, p)
        a = sqrt(gamma*p/rho)
        smax = max(smax, abs(uvel)+a, abs(vvel)+a)
      end do
    end do

    dxmin = minval(g%x%width(1:nc(1)))
    dymin = minval(g%y%width(1:nc(2)))

    dtloc = CFL * min(dxmin, dymin) / smax
  end function compute_dt_cfl

  !=========================================================
  pure function flux_lf(ul, ur, dir) result(fnum)
    real(rk), intent(in) :: ul(neq), ur(neq)
    integer, intent(in)  :: dir
    real(rk) :: fnum(neq)
    real(rk) :: fl(neq), fr(neq), alpha
    real(rk) :: rho_l, ux_l, uy_l, p_l, a_l
    real(rk) :: rho_r, ux_r, uy_r, p_r, a_r

    fl = flux_euler_2d(ul, dir)
    fr = flux_euler_2d(ur, dir)

    call cons_to_prim(ul, rho_l, ux_l, uy_l, p_l)
    call cons_to_prim(ur, rho_r, ux_r, uy_r, p_r)

    a_l = sqrt(gamma*p_l/rho_l)
    a_r = sqrt(gamma*p_r/rho_r)

    if (dir==1) then
      alpha = max(abs(ux_l)+a_l, abs(ux_r)+a_r)
    else
      alpha = max(abs(uy_l)+a_l, abs(uy_r)+a_r)
    end if

    fnum = 0.5_rk*(fl + fr) - 0.5_rk*alpha*(ur - ul)
  end function flux_lf

  !=========================================================
  pure function flux_euler_2d(v, dir) result(f)
    real(rk), intent(in) :: v(neq)
    integer, intent(in)  :: dir
    real(rk) :: f(neq)
    real(rk) :: rho, uvel, vvel, E, p

    rho = max(v(1), 1e-14_rk)
    uvel = v(2)/rho
    vvel = v(3)/rho
    E = v(4)
    p = (gamma-1.0_rk)*(E - 0.5_rk*rho*(uvel**2 + vvel**2))
    p = max(p, 1e-14_rk)

    if (dir==1) then
      f(1) = rho*uvel
      f(2) = rho*uvel**2 + p
      f(3) = rho*uvel*vvel
      f(4) = (E+p)*uvel
    else
      f(1) = rho*vvel
      f(2) = rho*uvel*vvel
      f(3) = rho*vvel**2 + p
      f(4) = (E+p)*vvel
    end if
  end function flux_euler_2d

  !=========================================================
 pure subroutine cons_to_prim(Uc, rho, uvel, vvel, p)
  real(rk), intent(in)  :: Uc(neq)
  real(rk), intent(out) :: rho, uvel, vvel, p
  real(rk) :: E

  rho  = max(Uc(1), 1e-14_rk)
  uvel = Uc(2)/rho
  vvel = Uc(3)/rho
  E    = Uc(4)
  p    = (gamma-1.0_rk)*(E - 0.5_rk*rho*(uvel*uvel + vvel*vvel))
  p    = max(p, 1e-14_rk)
end subroutine cons_to_prim

  !=========================================================
  pure function prim_to_cons(rho, uvel, vvel, p) result(Uc)
    real(rk), intent(in) :: rho, uvel, vvel, p
    real(rk) :: Uc(neq)
    real(rk) :: E
    E = p/(gamma-1.0_rk) + 0.5_rk*rho*(uvel*uvel + vvel*vvel)
    Uc = [rho, rho*uvel, rho*vvel, E]
  end function prim_to_cons

  !=========================================================
  subroutine init_double_mach(v)
    real(rk), intent(out) :: v(:)
    integer :: i,j
    real(rk) :: x,y
    real(rk) :: Uc(neq)

    do j = 1, nc(2)
      y = g%y%center(j)
      do i = 1, nc(1)
        x = g%x%center(i)
        call shock_state(x, y, 0.0_rk, Uc)
        call set_cell(v, i, j, Uc)
      end do
    end do
  end subroutine init_double_mach

  !=========================================================
  subroutine output2d(u, time)
    real(rk), intent(in) :: u(:)
    real(rk), intent(in) :: time
    character(*), parameter :: folder = "./resultats/double_mach/"
    integer :: i, j
    real(rk) :: rho, uvel, vvel, p
    real(rk) :: Uc(neq)
    integer :: funit_x, funit_u

    open(newunit=funit_x, file=folder//"x.txt", status="replace")
    write(funit_x,'(a)') "i j x y"
    do j=1,nc(2)
      do i=1,nc(1)
        write(funit_x,'(2i6,2es18.8e3)') i, j, g%x%center(i), g%y%center(j)
      end do
    end do
    close(funit_x)

    open(newunit=funit_u, file=folder//"u.txt", status="replace")
    write(funit_u,'(a)') "time i j rho u v p"
    do j=1,nc(2)
      do i=1,nc(1)
        call get_cell(u, i, j, Uc)
        call cons_to_prim(Uc, rho, uvel, vvel, p)
        write(funit_u,'(f10.5,2i6,4es18.8e3)') time, i, j, rho, uvel, vvel, p
      end do
    end do
    close(funit_u)
  end subroutine output2d

end program double_mach
