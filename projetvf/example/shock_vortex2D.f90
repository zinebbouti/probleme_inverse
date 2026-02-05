program euler_2d_shock_vortex
   use, intrinsic :: iso_fortran_env, only: stdout => output_unit
   use precisions,    only: rk
   use schema_temps, only: rktvd
   use schema_weno,  only: weno, BC_OUTFLOW, BC_REFLECTIVE
   use grille,       only: grid2
   implicit none

   integer, parameter :: neq = 4
   integer, parameter :: nc(2) = [251,100]
   real(rk) :: u(neq*product(nc))
   real(rk) :: dt, time, time_out
   integer :: ii, jj, nout

   type(grid2) :: gx2d
   type(weno)  :: myweno(2)
   type(rktvd) :: ode

   real(rk), parameter :: gamma = 1.4_rk

!=================================================
! Grille
!=================================================
   call gx2d%cartesian(0._rk,2._rk,0._rk,1._rk,nc)

!=================================================
! WENO
!=================================================
   myweno(1) = weno(nc(1),k=3,eps=1.e-10_rk,bc_type=BC_OUTFLOW)
   myweno(2) = weno(nc(2),k=3,eps=1.e-10_rk,bc_type=BC_REFLECTIVE)

!=================================================
! Conditions initiales
!=================================================
   do concurrent(ii=1:nc(1), jj=1:nc(2))
      call ic_shock_vortex( [gx2d%x%center(ii), gx2d%y%center(jj)], &
           u((jj-1)*nc(1)*neq + (ii-1)*neq + 1) )
   end do

!=================================================
! Temps
!=================================================
   ode  = rktvd(rhs,size(u),3)
   time = 0._rk
   dt   = 1.e-4_rk
   nout = 40

   call output2d(u,time)

   do ii=1,nout
      time_out = 0.35_rk*ii/nout
      call ode%integrate(u,time,time_out,dt)
      call output2d(u,time)
      write(stdout,'("t = ",f8.4)') time
   end do

contains
!=================================================
! RHS
!=================================================
subroutine rhs(t,v,vdot)
   real(rk),intent(in)  :: t, v(:)
   real(rk),intent(out) :: vdot(:)

   ! X direction
   real(rk) :: vlx(neq,nc(1)), vrx(neq,nc(1))
   real(rk) :: vlx_loc(nc(1)), vrx_loc(nc(1))
   real(rk) :: tempx(nc(1))

   ! Y direction
   real(rk) :: vly(neq,nc(2)), vry(neq,nc(2))
   real(rk) :: vly_loc(nc(2)), vry_loc(nc(2))
   real(rk) :: tempy(nc(2))

   real(rk) :: fx(neq,0:nc(1),1:nc(2))
   real(rk) :: fy(neq,0:nc(2),1:nc(1))
   real(rk) :: ul(neq), ur(neq)

   integer :: i,j,k

!---------------- X ----------------
   do j=1,nc(2)
      do k=1,neq
         do i=1,nc(1)
            tempx(i)=v((j-1)*nc(1)*neq+(i-1)*neq+k)
         end do
         call myweno(1)%reconstruct(tempx,vlx_loc,vrx_loc)
         vlx(k,:)=vlx_loc
         vrx(k,:)=vrx_loc
      end do

      do i=1,nc(1)-1
         ul=vrx(:,i); ur=vlx(:,i+1)
         fx(:,i,j)=flux_lf(ul,ur,1)
      end do

      fx(:,0,j)     = flux_euler(vlx(:,1),1)
      fx(:,nc(1),j) = flux_euler(vrx(:,nc(1)),1)
   end do

!---------------- Y ----------------
   do i=1,nc(1)
      do k=1,neq
         do j=1,nc(2)
            tempy(j)=v((j-1)*nc(1)*neq+(i-1)*neq+k)
         end do
         call myweno(2)%reconstruct(tempy,vly_loc,vry_loc)
         vly(k,:)=vly_loc
         vry(k,:)=vry_loc
      end do

      do j=1,nc(2)-1
         ul=vry(:,j); ur=vly(:,j+1)
         fy(:,j,i)=flux_lf(ul,ur,2)
      end do

      ul=vry(:,1); ul(3)=-ul(3)
      fy(:,0,i)=flux_euler(ul,2)

      ur=vly(:,nc(2)); ur(3)=-ur(3)
      fy(:,nc(2),i)=flux_euler(ur,2)
   end do

!---------------- Divergence ----------------
   do j=1,nc(2)
      do i=1,nc(1)
         do k=1,neq
            vdot((j-1)*nc(1)*neq+(i-1)*neq+k)= &
            -(fx(k,i,j)-fx(k,i-1,j))/gx2d%x%width(i) &
            -(fy(k,j,i)-fy(k,j-1,i))/gx2d%y%width(j)
         end do
      end do
   end do
end subroutine rhs

!=================================================
! Flux Lax–Friedrichs
!=================================================
pure function flux_lf(ul,ur,dir) result(f)
   real(rk),intent(in)::ul(neq),ur(neq)
   integer,intent(in)::dir
   real(rk)::f(neq),fl(neq),fr(neq)
   real(rk)::rho,u,v,p,a,alpha

   fl=flux_euler(ul,dir)
   fr=flux_euler(ur,dir)

   rho=ul(1)
   u=ul(2)/rho
   v=ul(3)/rho
   p=(gamma-1)*(ul(4)-0.5*rho*(u*u+v*v))
   a=sqrt(gamma*p/rho)

   if(dir==1)then
      alpha=abs(u)+a
   else
      alpha=abs(v)+a
   end if

   f=0.5*(fl+fr)-0.5*alpha*(ur-ul)
end function flux_lf

!=================================================
! Flux Euler
!=================================================
pure function flux_euler(v,dir) result(f)
   real(rk),intent(in)::v(neq)
   integer,intent(in)::dir
   real(rk)::f(neq),rho,u,vv,p,E

   rho=v(1)
   u=v(2)/rho
   vv=v(3)/rho
   E=v(4)
   p=(gamma-1)*(E-0.5*rho*(u*u+vv*vv))

   if(dir==1)then
      f=[rho*u, rho*u*u+p, rho*u*vv, (E+p)*u]
   else
      f=[rho*vv, rho*u*vv, rho*vv*vv+p, (E+p)*vv]
   end if
end function flux_euler

!=================================================
! Conditions initiales choc + vortex
!=================================================
pure subroutine ic_shock_vortex(x,U)
   real(rk),intent(in)  :: x(2)
   real(rk),intent(out) :: U(neq)

   real(rk) :: rhoL,uL,vL,pL, EL
   real(rk) :: rhoR,uR,vR,pR, ER
   real(rk) :: aL, M1
   real(rk) :: r,theta,tau,du,dv,dT, TT, C

   real(rk),parameter :: eps=0.3_rk, rc=0.05_rk, alpha=0.204_rk
   real(rk),parameter :: xc=0.25_rk, yc=0.5_rk

   ! ----------- Etat gauche (amont) -----------
   M1  = 1.1_rk
   rhoL = 1._rk
   pL   = 1._rk
   vL   = 0._rk
   aL   = sqrt(gamma*pL/rhoL)
   uL   = M1*aL

   ! ----------- Etat droit (aval) : choc normal -----------
   rhoR = rhoL * ((gamma+1._rk)*M1*M1) / ((gamma-1._rk)*M1*M1 + 2._rk)
   pR   = pL   * (1._rk + 2._rk*gamma/(gamma+1._rk) * (M1*M1 - 1._rk))
   vR   = 0._rk
   uR   = uL * rhoL / rhoR   ! conservation du flux de masse

   ! ----------- Base state selon x -----------
   if (x(1) < 0.5_rk) then
      ! Base = etat gauche
      ! ----------- Vortex surimposé à gauche -----------
      r = sqrt((x(1)-xc)**2 + (x(2)-yc)**2)
      if (r > 1.e-14_rk) then
         tau   = r/rc
         theta = atan2(x(2)-yc, x(1)-xc)

         du =  eps*tau*exp(alpha*(1._rk - tau*tau)) * sin(theta)
         dv = -eps*tau*exp(alpha*(1._rk - tau*tau)) * cos(theta)

         dT = -(gamma-1._rk)*eps*eps*exp(2._rk*alpha*(1._rk - tau*tau)) &
              /(4._rk*alpha*gamma)

         uL = uL + du
         vL = vL + dv

         ! S=0 => p/rho^gamma = constant = pL/rhoL^gamma
         C  = pL / (rhoL**gamma)
         TT = (pL/rhoL) + dT               ! T = T0 + T~
         rhoL = (TT/C)**(1._rk/(gamma-1._rk))
         pL   = C * rhoL**gamma
      end if

      EL = pL/(gamma-1._rk) + 0.5_rk*rhoL*(uL*uL + vL*vL)
      U  = [rhoL, rhoL*uL, rhoL*vL, EL]

   else
      ER = pR/(gamma-1._rk) + 0.5_rk*rhoR*(uR*uR + vR*vR)
      U  = [rhoR, rhoR*uR, rhoR*vR, ER]
   end if

end subroutine ic_shock_vortex


!=================================================
! Output
!=================================================
subroutine output2d(u,time)
   real(rk),intent(in)::u(:),time
   integer,save::fu=0
   integer::i,j
   real(rk)::rho,uvel,vvel,E,p

   if(fu==0)open(newunit=fu,file="resultats/shock_vortex/u.txt",status="replace")

   do j=1,nc(2)
      do i=1,nc(1)
         rho=u((j-1)*nc(1)*neq+(i-1)*neq+1)
         uvel=u((j-1)*nc(1)*neq+(i-1)*neq+2)/rho
         vvel=u((j-1)*nc(1)*neq+(i-1)*neq+3)/rho
         E=u((j-1)*nc(1)*neq+(i-1)*neq+4)
         p=(gamma-1)*(E-0.5*rho*(uvel*uvel+vvel*vvel))
         write(fu,'(f8.4,2i5,4es16.6)') time,i,j,rho,uvel,vvel,p
      end do
   end do
   write(fu,*)
end subroutine output2d

end program euler_2d_shock_vortex
