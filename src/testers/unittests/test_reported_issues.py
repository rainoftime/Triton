#!/usr/bin/env python2
# coding: utf-8
"""Test all reported issues."""

import unittest
from triton import *


class TestIssue602(unittest.TestCase):

    """Testing https://github.com/JonathanSalwan/Triton/issues/602"""

    def setUp(self):
        """Define the arch."""
        self.Triton = TritonContext()
        self.Triton.setArchitecture(ARCH.X86)

    def test_issue(self):
        code = [
            (0x080487D7, "8B450C".decode("hex")), # mov     eax, [ebp+0xC]
        ]
        self.ADDR = 0x90000000

        self.symvar = self.Triton.newSymbolicVariable(32)
        astC = self.Triton.getAstContext()
        symvar_ast = astC.variable(self.symvar)
        sevar = self.Triton.newSymbolicExpression(symvar_ast)
        self.Triton.assignSymbolicExpressionToRegister(sevar, self.Triton.registers.ebp)

        # Setting value at ADDR + 0xC
        self.Triton.setConcreteMemoryValue(self.ADDR + 0xC + 0, 0x11)
        self.Triton.setConcreteMemoryValue(self.ADDR + 0xC + 1, 0x22)
        self.Triton.setConcreteMemoryValue(self.ADDR + 0xC + 2, 0x33)
        self.Triton.setConcreteMemoryValue(self.ADDR + 0xC + 3, 0x44)

        self.Triton.addCallback(self.callback_concrete_mem_val, CALLBACK.GET_CONCRETE_MEMORY_VALUE)

        for (addr, opcode) in code:
            # Build an instruction
            inst = Instruction()

            # Setup opcode
            inst.setOpcode(opcode)

            # Setup Address
            inst.setAddress(addr)

            # Process everything
            self.Triton.processing(inst)

        self.assertEqual(self.Triton.getConcreteSymbolicVariableValue(self.symvar), 0x90000000)
        self.assertEqual(self.Triton.getConcreteRegisterValue(self.Triton.registers.eax), 0x44332211)


    def callback_concrete_mem_val(self, api, memacc):
        api.setConcreteSymbolicVariableValue(self.symvar, self.ADDR)

