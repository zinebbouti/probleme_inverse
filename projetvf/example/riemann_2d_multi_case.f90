program riemann_2d_multi_case
    use, intrinsic :: iso_fortran_env, only: stderr => error_unit, stdout => output_unit
    use precisions, only: rk
    use schema_temps, only: rktvd
    use schema_weno, only: weno, BC_PERIODIC, BC_OUTFLOW
    use grille, only: grid2
    implicit none

    integer, parameter :: neq = 4
    !integer, parameter :: nc(2) = [400, 400] !pour case 3 et 15
    integer, parameter :: nc(2) = [200, 200]  ! pour case 12
    !pour case 12 
    real(rk) :: u(neq*product(nc)), dt, time, time_end
    integer :: ii, jj, test_case
    type(grid2) :: gx2d
    type(weno) :: myweno(2)
    type(rktvd) :: ode
    real(rk), parameter :: gamma = 1.4_rk
    real(rk), dimension(neq) :: tmp
    character(len=50) :: case_description

    !========================
    ! CHOIX DU CAS
    !========================
    write(stdout, '(a)') repeat('=', 70)
    write(stdout, '(a)') 'CHOIX DU CAS DE RIEMANN 2D'
    write(stdout, '(a)') repeat('=', 70)
    write(stdout, '(a)') 'Cas disponibles:'
    write(stdout, '(a)') '  3  : Configuration S, S, S, S (4 shocks)'
    write(stdout, '(a)') '  12 : Configuration J, S, S, J (contacts + shocks)'
    write(stdout, '(a)') '  15 : Configuration J, R, S, J (contact + rarefaction + shock)'
    write(stdout, '(a)') repeat('-', 70)
    write(stdout, '(a)', advance='no') 'Entrez le numéro du cas (3, 12 ou 15): '
    read(*, *) test_case
    
    ! Validation
    if (test_case /= 3 .and. test_case /= 12 .and. test_case /= 15) then
        write(stderr, '(a)') 'Erreur: Cas invalide. Choisissez 3, 12 ou 15.'
        stop
    end if
    
    ! Description du cas
    select case(test_case)
        case(3)
            case_description = 'Case 3: S, S, S, S (4 shocks)'
        case(12)
            case_description = 'Case 12: J, S, S, J (contacts + shocks)'
        case(15)
            case_description = 'Case 15: J, R, S, J (mixed waves)'
    end select

    write(stdout, '(a)') repeat('=', 70)
    write(stdout, '(a)') '2D RIEMANN PROBLEM'
    write(stdout, '(a)') 'Reference: Liska & Wendroff, SIAM J. Sci. Comput. (2003)'
    write(stdout, '(a,a)') 'Configuration: ', trim(case_description)
    write(stdout, '(a)') repeat('=', 70)
    write(stdout, '(a,i4,a,i4)') ' Grille: ', nc(1), ' × ', nc(2)
    write(stdout, '(a)') ' Domaine: [0,1] × [0,1]'
    write(stdout, '(a)') ' Schéma: WENO5 + RK3-TVD + Lax-Friedrichs'
    write(stdout, '(a)') repeat('=', 70)

    !========================
    ! Grille 2D
    !========================
    call gx2d%cartesian(xmin=0.0_rk, xmax=1.0_rk, ymin=0.0_rk, ymax=1.0_rk, ncells=nc)
    
    !========================
    ! Initialisation WENO
    !========================
    myweno(1) = weno(ncells=nc(1), k=3, eps=1e-6_rk,bc_type=BC_OUTFLOW)
    myweno(2) = weno(ncells=nc(2), k=3, eps=1e-6_rk,bc_type=BC_OUTFLOW)

    !========================
    ! Conditions initiales (4 quadrants)
    !========================
    write(stdout, '(a)') 'Initialisation des 4 quadrants...'
    do jj = 1, nc(2)
        do ii = 1, nc(1)
            call ic([gx2d%x%center(ii), gx2d%y%center(jj)], tmp, test_case)
            u((jj-1)*nc(1)*neq + (ii-1)*neq + 1 : (jj-1)*nc(1)*neq + ii*neq) = tmp
        end do
    end do
    write(stdout, '(a)') '  ✓ OK'

    !========================
    ! Intégration temporelle
    !========================
    ode = rktvd(rhs, size(u), 3)
    time = 0.0_rk
    time_end = 0.25_rk
    dt = 5e-5_rk
    !dt = 3.0e-4_rk  ! Valeur pour case 3

    write(stdout, '(a)') repeat('-', 70)
    write(stdout, '(a,f6.3)') ' Temps final: t = ', time_end
    write(stdout, '(a,es10.3)') ' dt initial: ', dt
    write(stdout, '(a)') repeat('-', 70)

    ! Snapshot initial
    call output2d(u, time, test_case)

    ! Intégration
    call ode%integrate(u, time, time_end, dt)
    
    write(stdout, '(a)') repeat('=', 70)
    write(stdout, '(a,f8.5)') ' ✓ Simulation terminée à t = ', time
    write(stdout, '(a)') repeat('=', 70)
    
    ! Snapshot final
    call output2d(u, time, test_case)

contains

    !========================
    ! Conditions initiales : Sélection du cas
    !========================
    subroutine ic(x, uvec, case_num)
        real(rk), intent(in) :: x(2)
        real(rk), intent(out) :: uvec(neq)
        integer, intent(in) :: case_num
        real(rk) :: rho, u_vel, v_vel, p, E
        
        select case(case_num)
            case(3)
                call ic_case3(x, rho, u_vel, v_vel, p)
            case(12)
                call ic_case12(x, rho, u_vel, v_vel, p)
            case(15)
                call ic_case15(x, rho, u_vel, v_vel, p)
        end select

        E = p/(gamma-1.0_rk) + 0.5_rk*rho*(u_vel**2 + v_vel**2)
        uvec = [rho, rho*u_vel, rho*v_vel, E]
    end subroutine ic

    !========================
    ! Case 3: S, S, S, S (4 shocks)
    !========================
    subroutine ic_case3(x, rho, u, v, p)
    real(rk), intent(in) :: x(2)
    real(rk), intent(out) :: rho, u, v, p
    
    if (x(2) > 0.5_rk) then           ! y > 0.5 (HAUT)
        if (x(1) < 0.5_rk) then       ! x < 0.5 (GAUCHE)
            ! Haut-gauche (upper-left)
            p = 0.3_rk
            rho = 0.5323_rk
            u = 1.206_rk
            v = 0.0_rk
        else                          ! x >= 0.5 (DROIT)
            ! Haut-droit (upper-right)
            p = 1.5_rk
            rho = 1.5_rk
            u = 0.0_rk
            v = 0.0_rk
        end if
    else                              ! y <= 0.5 (BAS)
        if (x(1) < 0.5_rk) then       ! x < 0.5 (GAUCHE)
            ! Bas-gauche (lower-left)
            p = 0.029_rk
            rho = 0.138_rk
            u = 1.206_rk
            v = 1.206_rk
        else                          ! x >= 0.5 (DROIT)
            ! Bas-droit (lower-right)
            p = 0.3_rk
            rho = 0.5323_rk
            u = 0.0_rk
            v = 1.206_rk
        end if
    end if
end subroutine ic_case3

    !========================
    ! Case 12: J, S, S, J (contacts + shocks)
    !========================
    subroutine ic_case12(x, rho, u, v, p)
        real(rk), intent(in) :: x(2)
        real(rk), intent(out) :: rho, u, v, p
        
        if (x(2) > 0.5_rk) then
            if (x(1) < 0.5_rk) then
                ! Haut-gauche
                rho = 1.0_rk
                u = 0.7276_rk
                v = 0.0_rk
                p = 1.0_rk
            else
                ! Haut-droit
                rho = 0.5313_rk
                u = 0.0_rk
                v = 0.0_rk
                p = 0.4_rk
            end if
        else
            if (x(1) < 0.5_rk) then
                ! Bas-gauche
                rho = 0.8_rk
                u = 0.0_rk
                v = 0.0_rk
                p = 1.0_rk
            else
                ! Bas-droit
                rho = 1.0_rk
                u = 0.0_rk
                v = 0.7276_rk
                p = 1.0_rk
            end if
        end if
    end subroutine ic_case12
subroutine ic_case15(x, rho, u, v, p)
    real(rk), intent(in) :: x(2)
    real(rk), intent(out) :: rho, u, v, p
    
    if (x(2) > 0.5_rk) then
        if (x(1) < 0.5_rk) then
            ! Haut-gauche
            rho = 0.5197_rk
            u = -0.6259_rk
            v = -0.3_rk
            p = 0.4_rk
        else
            ! Haut-droit
            rho = 1.0_rk
            u = 0.1_rk
            v = -0.3_rk
            p = 1.0_rk
        end if
    else
        if (x(1) < 0.5_rk) then
            ! Bas-gauche
            rho = 0.8_rk
            u = 0.1_rk
            v = -0.3_rk
            p = 0.4_rk
        else
            ! Bas-droit
            rho = 0.5313_rk
            u = 0.1_rk
            v = 0.4276_rk
            p = 0.4_rk
        end if
    end if
end subroutine ic_case15

    !========================
    ! RHS avec WENO5 2D
    !========================
    subroutine rhs(t, v, vdot)
        real(rk), intent(in) :: t
        real(rk), intent(in) :: v(:)
        real(rk), intent(out) :: vdot(:)
        real(rk) :: fedges_x(neq, 0:nc(1), 1:nc(2))
        real(rk) :: fedges_y(neq, 0:nc(2), 1:nc(1))
        real(rk) :: vl_x(neq, nc(1)), vr_x(neq, nc(1))
        real(rk) :: vl_y(neq, nc(2)), vr_y(neq, nc(2))
        real(rk) :: vcomp_x(nc(1)), vl_local_x(nc(1)), vr_local_x(nc(1))
        real(rk) :: vcomp_y(nc(2)), vl_local_y(nc(2)), vr_local_y(nc(2))
        real(rk) :: ul(neq), ur(neq)
        integer :: i, j, k

        vdot = 0.0_rk

        !========================
        ! Direction X
        !========================
        do j = 1, nc(2)
            do k = 1, neq
                do i = 1, nc(1)
                    vcomp_x(i) = v((j-1)*nc(1)*neq + (i-1)*neq + k)
                end do
                
                call myweno(1)%reconstruct(vcomp_x, vl_local_x, vr_local_x)
                
                vl_x(k, :) = vl_local_x
                vr_x(k, :) = vr_local_x
            end do

            do i = 1, nc(1)-1
                ul = vr_x(:, i)
                ur = vl_x(:, i+1)
                fedges_x(:, i, j) = flux_lax_friedrichs_2d(ul, ur, gamma, 1)
            end do
            
            fedges_x(:, 0, j) = flux_lax_friedrichs_2d(vl_x(:,1), vl_x(:,1), gamma, 1)
            fedges_x(:, nc(1), j) = flux_lax_friedrichs_2d(vr_x(:,nc(1)), vr_x(:,nc(1)), gamma, 1)
        end do

        !========================
        ! Direction Y
        !========================
        do i = 1, nc(1)
            do k = 1, neq
                do j = 1, nc(2)
                    vcomp_y(j) = v((j-1)*nc(1)*neq + (i-1)*neq + k)
                end do
                
                call myweno(2)%reconstruct(vcomp_y, vl_local_y, vr_local_y)
                
                vl_y(k, :) = vl_local_y
                vr_y(k, :) = vr_local_y
            end do

            do j = 1, nc(2)-1
                ul = vr_y(:, j)
                ur = vl_y(:, j+1)
                fedges_y(:, j, i) = flux_lax_friedrichs_2d(ul, ur, gamma, 2)
            end do
            
            fedges_y(:, 0, i) = flux_lax_friedrichs_2d(vl_y(:,1), vl_y(:,1), gamma, 2)
            fedges_y(:, nc(2), i) = flux_lax_friedrichs_2d(vr_y(:,nc(2)), vr_y(:,nc(2)), gamma, 2)
        end do

        !========================
        ! Divergence des flux
        !========================
        do j = 1, nc(2)
            do i = 1, nc(1)
                do k = 1, neq
                    vdot((j-1)*nc(1)*neq + (i-1)*neq + k) = &
                        -(fedges_x(k,i,j) - fedges_x(k,i-1,j))/gx2d%x%width(i) &
                        -(fedges_y(k,j,i) - fedges_y(k,j-1,i))/gx2d%y%width(j)
                end do
            end do
        end do
    end subroutine rhs

    !========================
    ! Flux de Lax-Friedrichs 2D
    !========================
    pure function flux_lax_friedrichs_2d(ul, ur, gamma, dir) result(fnum)
        real(rk), intent(in) :: ul(neq), ur(neq)
        real(rk), intent(in) :: gamma
        integer, intent(in) :: dir
        real(rk) :: fnum(neq)
        real(rk) :: fl(neq), fr(neq), alpha
        real(rk) :: rho_l, u_l, v_l, p_l, a_l
        real(rk) :: rho_r, u_r, v_r, p_r, a_r
        real(rk), parameter :: eps = 1.0e-10_rk

        fl = flux_euler_2d(ul, gamma, dir)
        fr = flux_euler_2d(ur, gamma, dir)

        rho_l = max(ul(1), eps)
        u_l = ul(2)/rho_l
        v_l = ul(3)/rho_l
        p_l = (gamma-1.0_rk)*(ul(4) - 0.5_rk*rho_l*(u_l**2 + v_l**2))
        p_l = max(p_l, eps)
        a_l = sqrt(gamma*p_l/rho_l)

        rho_r = max(ur(1), eps)
        u_r = ur(2)/rho_r
        v_r = ur(3)/rho_r
        p_r = (gamma-1.0_rk)*(ur(4) - 0.5_rk*rho_r*(u_r**2 + v_r**2))
        p_r = max(p_r, eps)
        a_r = sqrt(gamma*p_r/rho_r)

        if (dir==1) then
            alpha = max(abs(u_l)+a_l, abs(u_r)+a_r)
        else
            alpha = max(abs(v_l)+a_l, abs(v_r)+a_r)
        end if

        fnum = 0.5_rk*(fl + fr) - 0.5_rk*alpha*(ur - ul)
    end function flux_lax_friedrichs_2d

    !========================
    ! Flux d'Euler 2D
    !========================
    pure function flux_euler_2d(v, gamma, dir) result(f)
        real(rk), intent(in) :: v(neq)
        real(rk), intent(in) :: gamma
        integer, intent(in) :: dir
        real(rk) :: f(neq)
        real(rk) :: rho, u, vv, E, p
        real(rk), parameter :: eps = 1.0e-10_rk

        rho = max(v(1), eps)
        u = v(2)/rho
        vv = v(3)/rho
        E = v(4)
        p = (gamma-1.0_rk)*(E - 0.5_rk*rho*(u**2 + vv**2))
        p = max(p, eps)

        if (dir==1) then
            f = [rho*u, rho*u**2 + p, rho*u*vv, (E+p)*u]
        else
            f = [rho*vv, rho*u*vv, rho*vv**2 + p, (E+p)*vv]
        end if
    end function flux_euler_2d

    !========================
    ! Output 2D
    !========================
    subroutine output2d(u, time, case_num)
        implicit none
        real(rk), intent(in) :: u(:)
        real(rk), intent(in) :: time
        integer, intent(in) :: case_num
        character(*), parameter :: folder = "./resultats/riemann/"
        character(len=256) :: filename
        integer :: funit, i, j
        real(rk) :: rho, u_vel, v_vel, p

        write(filename, '(a,a,i0,a,f0.6,a)') folder, "case", case_num, "_t", time, ".txt"
        
        open(newunit=funit, file=trim(filename), status="replace")
        write(funit, '(a)') "x,y,rho,u,v,p"
        
        do j = 1, nc(2)
            do i = 1, nc(1)
                rho = u((j-1)*nc(1)*neq + (i-1)*neq + 1)
                u_vel = u((j-1)*nc(1)*neq + (i-1)*neq + 2)/rho
                v_vel = u((j-1)*nc(1)*neq + (i-1)*neq + 3)/rho
                p = (gamma-1.0_rk)*(u((j-1)*nc(1)*neq + (i-1)*neq + 4) - &
                    0.5_rk*rho*(u_vel**2 + v_vel**2))
                
                write(funit, '(6(es15.7,a))') &
                    gx2d%x%center(i), ',', gx2d%y%center(j), ',', &
                    rho, ',', u_vel, ',', v_vel, ',', p
            end do
        end do
        
        close(funit)
        write(stdout, '(a,a)') '  ✓ Sauvegardé: ', trim(filename)
    end subroutine output2d

end program riemann_2d_multi_case