module precisions
!! Module définissant la précision réelle utilisée dans tout le code
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
   private

   ! Précision réelle globale
   public :: rk
   integer, parameter :: rk = real64

end module precisions
