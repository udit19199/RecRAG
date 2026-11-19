Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 2 items.

=== ITEM id=785b7b12c0d1cfc0 ===
Gold sentences and facts:
Sentence 0: In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Sorcerer builds for season 2.
  Fact 0.0: The guide is a Diablo 4 guide.
  Fact 0.1: The guide gathers the best Diablo 4 Sorcerer builds for season 2.
  Fact 0.2: The guide simplifies the best Diablo 4 Sorcerer builds for season 2.
Sentence 1: In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Barbarian builds for season 2.
  Fact 1.0: The guide is a Diablo 4 guide.
  Fact 1.1: The guide gathers the best Diablo 4 Barbarian builds for season 2.
  Fact 1.2: The guide simplifies the best Diablo 4 Barbarian builds for season 2.

Retrieved items:
<item 1>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
Ring 2: Aspect of Limitless Rage

If you want to get into the nitty gritty of stat priorities, refer to AaronActionRPG’s original guide.

Paragon Boards for Double Swing Barbarian build

The Paragon Board system is very complex, and you’ll want to carefully pair your Glyphs with specific boards for your class:

Starter Board / Bloodfeeder Glyph Decimator / Ambidextrous Glyph Carnage / Might Glyph Flawless Technique / Ire Glyph Weapon Master / Territorial Glyph Bone
Breaker / Exploit Glyph Warbringer / Wrath Glyph

You should level your Glyphs to 15 in this specific order:

Exploit Territorial Ambidextrous Might Wrath Bloodfeeder Ire

You can path your own way through the Paragon Boards following these basic guidelines. However, for the best results, follow AaronActionRPG’s path exactly.

How to make a great Barbarian build

Barbarians are Diablo 4’s only pure melee class, with the other classes having hybrid builds for multiple ranges.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Barbarian
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Barbarian
- Wildheart -PART_OF-> Barbarian
- Berserker -PART_OF-> Barbarian
- Wild Magic -PART_OF-> Barbarian

Entities: Barbarian, Paragon Board system, Glyphs, AaronActionRPG, The best Barbarian builds and skills in Diablo 4, Double Swing Barbarian build, Aspect of
Limitless Rage, Ring 2, Starter Board, Bloodfeeder Glyph, Decimator, Ambidextrous Glyph, Carnage, Might Glyph, Flawless Technique
</item 1>

<item 2>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
As such, you’ll need to unlock new skills before enhancing existing ones:

Frenzy / Enhanced Frenzy Hammer of the Ancients / Enhanced Hammer of the Ancients / Furious Hammer of the Ancients Combat Frenzy Rallying Cry / Enhanced
Rallying Cry / Tactical Rallying Cry Hammer of the Ancients (ranks 2-3) Leap / Enhanced Leap / Power Leap Hammer of the Ancients (ranks 4-5) Death Blow /
Enhanced Death Blow / Warrior’s Death Blow / Death Blow (rank 2-5) Wrath of the Berserker / Prime Wrath of the Berserker / Supreme Wrath of the Berserker Pit
Fighter (ranks 1-3) Tempered Fury Furious Impulse (ranks 1-3) Walking Arsenal Hamstring No Mercy Expose Vulnerability (ranks 1-3) No Mercy (ranks 2-3) Swiftness
(ranks 1-3) Thick Skin (ranks 1-3) Counteroffensive (ranks 1-3) Defensive

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Barbarian
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Barbarian
- Wildheart -PART_OF-> Barbarian
- Berserker -PART_OF-> Barbarian
- Wild Magic -PART_OF-> Barbarian

Entities: Barbarian, Hammer of the Ancients, Frenzy, Death Blow, Rallying Cry, Leap, Wrath of the Berserker, Combat Frenzy, Pit Fighter, Tempered Fury, Furious
Impulse, Walking Arsenal, No Mercy, Expose Vulnerability, Swiftness
</item 2>

<item 3>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
If you don’t want to end up dead more often than not, make sure to invest heavily in defensive skills while making your way toward one of the six Key Passives
we listed above.

Looking for other Diablo 4 builds? We have explainers on the Barbarian, Rogue, Druid, and Necromancer classes.

Graph facts:
- The best Sorcerer builds and skills in Diablo 4 -MENTIONS-> Enchantment
- The best Sorcerer builds and skills in Diablo 4 -MENTIONS-> Sanctuary
- The best Sorcerer builds and skills in Diablo 4 -MENTIONS-> Sorcerer builds
- The best Sorcerer builds and skills in Diablo 4 -MENTIONS-> lightning
- The best Sorcerer builds and skills in Diablo 4 -MENTIONS-> ice

Entities: The best Sorcerer builds and skills in Diablo 4, Sorcerer builds and skills, defensive skills, six Key Passives, Barbarian class, Rogue class, Druid
class, Necromancer class, Diablo 4, Polygon
</item 3>

<item 4>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
The idea here is that you’re constantly swapping between your skills and weapons to give yourself massive damage buffs.

When making a build, you want to have one of these Key Passives in mind and choose your skills accordingly. Thankfully skills that share certain aspects (like
Bleed) are color-coded on the skill tree, and you can even use the in-game search function to find them all. Other Barbarian builds let you take advantage of
mechanics like Thorns or Shouts. But they still typically end with one of these Key Passives.

Looking for other Diablo 4 builds? We have explainers on the Sorcerer, Rogue, Druid, and Necromancer classes.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Barbarian
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Barbarian
- Wildheart -PART_OF-> Barbarian
- Berserker -PART_OF-> Barbarian
- Wild Magic -PART_OF-> Barbarian

Entities: Barbarian, Bleed, The best Barbarian builds and skills in Diablo 4, Key Passives, Thorns, Shouts, Sorcerer, Rogue, Druid, Necromancer, Diablo 4,
Polygon
</item 4>

<item 5>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
Barbarians are a relatively simple class in Diablo 4: They smash stuff. However, Diablo 4 is underscored by math that’s so complex that even the simple act of
smashing stuff becomes a dizzying equation. That’s where we come in, with our suggestions of the best Barbarian builds for Diablo 4 season 2.

Barbarians are big, muscular, and powerful. They excel at using Bleed to kill their enemies over time. And their unique weapon-swapping system means they get to
have more Legendary powers at their disposal than any other class.

In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Barbarian builds for season 2. If you need even more information on item affixes or
other more complex ideas, we’ve been sure to link out to the more detailed versions of these builds, as well as where to find their original creators.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Barbarian
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Barbarian
- Wildheart -PART_OF-> Barbarian
- Berserker -PART_OF-> Barbarian
- Wild Magic -PART_OF-> Barbarian

Entities: Barbarian, item affixes, Barbarian builds, Bleed, weapon-swapping system, Legendary powers, season 2, The best Barbarian builds and skills in Diablo
4, Diablo 4, Polygon
</item 5>

<item 6>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
dealing X% of its damage to enemies

(Sunken Ruins dungeon in Scosglen) — Hammer of the Ancients quakes outwards, dealing X% of its damage to enemies Aspect of the Expectant (Underroot dungeon in
Scosglen) — Attacking enemies with a Basic Skill increases the damage of your next Core Skill cast by X% up to X%

(Underroot dungeon in Scosglen) — Attacking enemies with a Basic Skill increases the damage of your next Core Skill cast by X% up to X% Edgemaster’s Aspect
(Oldstones dungeon in Scosglen) — Skills deal X% increased damage based on Primary Resource when cast, receiving the maximum benefit while you have full Primary
Resource

Barbarian Season 2 endgame build — Double Swing

This build is based off of AaronActionRPG’s Double Swing Barbarian endgame guide at IcyVeins, and is a general endgame guide capable of finding success in all
activities.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Barbarian
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Barbarian
- Wildheart -PART_OF-> Barbarian
- Berserker -PART_OF-> Barbarian
- Wild Magic -PART_OF-> Barbarian

Entities: Barbarian, IcyVeins, Hammer of the Ancients, Sunken Ruins dungeon, Scosglen, Aspect of the Expectant, Underroot dungeon, Edgemaster’s Aspect,
Oldstones dungeon, Barbarian Season 2 endgame build — Double Swing, AaronActionRPG, Double Swing Barbarian endgame guide, The best Barbarian builds and skills
in Diablo 4, Diablo 4, Polygon
</item 6>

<item 7>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Sorcerers are the magic-slinging wizards of Sanctuary in Diablo 4. But with each element they can harness bringing its own unique effect, it can be easy to get
overwhelmed when trying to design a Sorcerer build. That’s where we come in, with our suggestions of the best Sorcerer builds for Diablo 4 season 2.

Sorcerers are capable of harnessing lightning to deal with crowds, ice to keep the hordes at bay, and fire to burn the demons of hell. But they’re also able to
enchant their own spells with unique effects, intricately combining abilities for devastating results.

In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Sorcerer builds for season 2. If you need even more information on item affixes or other
more complex ideas, we’ve been sure to link out to the more detailed versions of these builds, as well as where to find their original creators.

Graph facts:
- 10 scorching-hot discoveries made about the sun in 2023 -MENTIONS-> lightning
- 54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods -MENTIONS-> lightning
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> lightning
- Diablo 4 season 2’s biggest draw is its quality-of-life updates -MENTIONS-> Sanctuary
- The best Barbarian builds and skills in Diablo 4 -MENTIONS-> item affixes

Entities: Sanctuary, Sorcerer builds, lightning, ice, fire, item affixes, The best Sorcerer builds and skills in Diablo 4, Sorcerer, Diablo 4, Polygon, season 2
</item 7>

<item 8>
Graph query row:
sorcererGuide: The best Sorcerer builds and skills in Diablo 4
sorcererPublishedOn: 2023-10-18T21:43:17+00:00
barbarianGuide: The best Barbarian builds and skills in Diablo 4
barbarianPublishedOn: 2023-10-19T17:16:31+00:00
sharedEntities: ['item affixes', 'Lexyu', 'IcyVeins', 'imprint system', 'Vampiric Powers', 'Pacts', 'armor', 'Edgemaster’s', 'Paragon Board system', 'Glyphs',
'Sorcerer', 'Druid', 'Necromancer', 'Diablo 4']
directConnections: []
sorcererEvidence: [{'page': 'The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)', 'passage': 'The best Sorcerer builds and
skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)\nSorcerers are the magic-slinging wizards of Sanctuary in Diablo 4. But with each element they can
harness bringing its own unique effect, it can be easy to get overwhelmed when trying to design a Sorcerer build. That’s where we come in, with our suggestions
of the best Sorcerer builds for Diablo 4 season 2.\n\nSorcerers are capable of harnessing lightning to deal with crowds, ice to keep the hordes at bay, and fire
to burn the demons of hell. But they’re also able to enchant their own spells with unique effects, intricately combining abilities for devastating
results.\n\nIn this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Sorcerer builds for season 2. If you need even more information on item
affixes or other more complex ideas, we’ve been sure to link out to the more detailed versions of these builds, as well as where to find their original
creators.'}, {'page': 'The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)', 'passage': 'The best Sorcerer builds and skills in
Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)\nBest Sorcerer leveling build\n\nThis build is based off Lexyu’s Sorcerer leveling guide at IcyVeins, and is
purely for leveling: i.e., allocating your 58 skill points between levels 1 and 50. If you want more detailed information about the item affixes you want to
look for on your gear, or which other Legendary powers are great for your build, check out the original IcyVeins guide.\n\nBest skills for leveling\n\nThis
Sorcerer build guide is going to see you harness the awesome power of Lightning to speed through the leveling process. There are a few options for Core skills
here, but we’re specifically going to be showing you Lexyu’s Chain Lightning variation. In this build, you’ll also use Frost Bolt, Teleport, Frost Nova,
Lightning Spear, and Ice Armor.\n\nHere are the best skills for a Sorcerer leveling build, organized by the exact order you should purchase them in — bearing in
mind that some skills require you to allocate a number of skill points in the tree in order to unlock them. As such,'}, {'page': 'The best Sorcerer builds and
skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)', 'passage': 'The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)\nAs
such, you’ll need to unlock new skills before enhancing existing ones:\n\nFrost Bolt / Enhanced Frost Bolt Chain Lightning Glinting Frost Bolt Enhanced Chain
Lightning / Greater Chain Lightning Teleport Frost Nova / Enhanced Frost Nova / Mystical Frost Nova Chain Lightning (rank 2-5) Fireball (rank 1 only for the
Enchantment slot) Lightning Spear Glass Cannon (rank 1-3) Devastation Elemental Dominance (rank 1-3) Coursing Currents (rank 1-3) Elemental Attunement Ice Armor
/ Enhanced Ice Armor / Shimmering Ice Armor Enhanced Teleport / Shimmering Teleport Enhanced Lightning Spear Vyr’s Mastery Invoked Lightning Spear / Lightning
Spear (rank 2-5) Frost Bolt (rank 2-5) Conduction (rank 1-3) Electrocution (rank 1-3) Align the Elements Protection (rank 1-3) Icy Veil (rank 1-3) Frost Nova
(rank 2-3)\n\nBest Enchantment setup for leveling\n\nThe Enchantment'}, {'page': 'The best Sorcerer builds and skills in Diablo 4 (Polygon,
2023-10-18T21:43:17+00:00)', 'passage': 'The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)\nFrost Nova (rank 2-3)\n\nBest
Enchantment setup for leveling\n\nThe Enchantment slot is key to a ton of your damage as a Sorcerer. You’ll gain two slots as you level, here’s which spell you
want to put in each slot:\n\nLevel 15: Fireball\n\nLevel 30: Chain Lightning\n\nBest Legendary Aspects for Sorcerer leveling build\n\nWhen you first start your
new character, complete these three dungeons first in order to collect their Aspects. Then place them on a piece of gear (ideally jewelry, when possible). Just
keep in mind that you should replace these Legendaries with new versions using the imprint system every few levels.'}, {'page': 'The best Sorcerer builds and
skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)', 'passage': 'The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)\nThis
guide assumes that you’re over level 50, have access to all 58 skill points, and are making progress on your Paragon Boards. If you want more detailed
information about the item affixes you want to look for on your gear, or why these Legendary powers are great for your build, check out the original MaxRoll
guide.\n\nBest skill points for Sorcerer Firewall endgame build\n\nThis build utilizes Firewall as its core skill and main means of dealing damage. It also uses
Fire Bolt, Inferno, Teleport, Ice Armor, and Flame Shield.\n\nSince you need all 58 points for this build, the order you select these skills in doesn’t matter
as long as you have the pre-requisite points in the earlier Skill Tree tiers.'}, {'page': 'The best Sorcerer builds and skills in Diablo 4 (Polygon,
2023-10-18T21:43:17+00:00)', 'passage': 'The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)\nHere is how you should spend your
Skill Points for this build:\n\nFire Bolt / Enhanced Fire Bolt / Glinting Fire Bolt\n\nFireball / Enhanced Fireball / Greater Fireball\n\nPotent Warding (rank
1-3)\n\nDevastation\n\nElemental Dominance (rank 1-3)\n\nFlame Shield / Enhanced Flame Shield / Shimmering Flame Shield\n\nTeleport (rank 1-4) / Enhanced
Teleport / Shimmering Teleport\n\nIce Armor/Enhanced Ice Armor\n\nGlass Cannon (rank 1-3)\n\nPrecision Magic (rank 1-3)\n\nAlign the Elements\n\nMana Shield
(rank 1-3)\n\nProtection (rank 1-3)\n\nFirewall (rank 1-5) / Enhanced Firewall / Wizard’s Firewall\n\nInner Flames (rank 1-3)\n\nCrippling Flames (rank
1-3)\n\nInferno/Prime Inferno/Supreme Inferno\n\nFiery Surge\n\nEndless Pyre\n\nCombustion\n\nEnhancement slots for Sorcerer Firewall build\n\nFor this
build,'}, {'page': 'The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)', 'passage': 'The best Sorcerer builds and skills in
Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)\nyou’ll want to use these spells for your two Enchantment Slots:\n\nFirewall Enchantment\n\nFireball\n\nVampiric
Powers for Sorcerer Firewall build\n\nSeason 2 adds the new Vampiric Powers, which allow you to slot five additional abilities into your character to give you
new effects. There are 22 total Vampiric Powers, and each are class agnostic, meaning they’re not unique to any one class.\n\nNorthwar recommends these five
Vampiric Powers:\n\nAnticipation\n\nPrey on the Weak\n\nRavenous\n\nFlowing Veins\n\nMetamorphosis\n\nTo slot those powers you’ll need to have these Pacts
available on your armor:\n\n9 Ferocity\n\n5 Divinity\n\n4 Eternity\n\nLegendary Powers and Uniques for Sorcerer Firewall build\n\nTo make this build work,
you’re going to need powerful Legendary Powers and Unique items to augment your build.'}, {'page': 'The best Sorcerer builds and skills in Diablo 4 (Polygon,
2023-10-18T21:43:17+00:00)', 'passage': 'The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)\nNorthwar recommends these
items:\n\nHelm : Shared Misery (dungeon drop) or Godslayer Crown (Unique)\n\n: Shared Misery (dungeon drop) or Godslayer Crown (Unique) Chest : Snowveiled
(dungeon drop) or Raiment of the Infinite (Unique)\n\n: Snowveiled (dungeon drop) or Raiment of the Infinite (Unique) Gloves : Edgemaster’s (dungeon drop),
Conceited, or Storm Swell (dungeon drop)\n\n: Edgemaster’s (dungeon drop), Conceited, or Storm Swell (dungeon drop) Pants : Disobedience (dungeon drop)\n\n:
Disobedience (dungeon drop) Boots : Ghostwalker (dungeon drop), Bounding Conduit (dungeon drop), Flickerstep (Unique)\n\n: Ghostwalker (dungeon drop), Bounding
Conduit (dungeon drop), Flickerstep (Unique) Wand : Edgemaster’s (dungeon drop), Conceited,'}, {'page': 'The best Sorcerer builds and skills in Diablo 4
(Polygon, 2023-10-18T21:43:17+00:00)', 'passage': 'The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)\nConceited, or Storm
Swell (dungeon drop)\n\n: Edgemaster’s (dungeon drop), Conceited, or Storm Swell (dungeon drop) Focus : Edgemaster’s (dungeon drop), Conceited, or Storm Swell
(dungeon drop)\n\n: Edgemaster’s (dungeon drop), Conceited, or Storm Swell (dungeon drop) Amulet : Control (dungeon drop)\n\n: Control (dungeon drop) Ring 1 :
Prodigy’s (dungeon drop) or Retribution (dungeon drop)\n\n: Prodigy’s (dungeon drop) or Retribution (dungeon drop) Ring 2: Engulfing Flames or X’Fal’s Corroded
Signet (Unique)\n\nIf you want to get into the nitty gritty of stat priorities, refer to Northwar’s original guide.'}, {'page': 'The best Sorcerer builds and
skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)', 'passage': 'The best Sorcerer builds and skills in Diablo 4 (Polygon,
2023-10-18T21:43:17+00:00)\nrefer to Northwar’s original guide.\n\nParagon Boards for Sorcerer Firewall build\n\nThe Paragon Board system is very complex, and
you’ll want to carefully pair your Glyphs with specific boards for your class:\n\nStarter Board / Reinforced Glyph Burning Instinct / Destruction Glyph Searing
Heat / Adept Glyph Static Surge / Control Glyph Enchantment Master / Exploit\n\nYou should level your Glyphs to 15 in this specific order:\n\nDestruction
Control Adept Exploit Reinforced\n\nYou can path your own way through the Paragon Boards following these basic guidelines. However, for the best results, follow
Northwar’s path exactly.\n\nHow to make a great Sorcerer build\n\nSorcerers are Diablo 4’s pure caster class, and use elemental magic to differentiate from the
spells of the Druid and Necromancer. The Sorcerer has six Key Passives, but only three elements, which is what you’ll base your build
around:\n\nLightning\n\nFire\n\nIce\n\nLightning builds utilize Lightning skills to deal with large groups of enemies at once.'}, {'page': 'The best Sorcerer
builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)', 'passage': 'The best Sorcerer builds and skills in Diablo 4 (Polygon,
2023-10-18T21:43:17+00:00)\nWhen making a Lightning build, you’ll end up using the Overflowing Energy Key Passive or the Vyr’s Master Key Passive, depending on
if you’re in melee range or not.\n\nIce builds specialize in slowing enemies down, making it easier to survive their attacks. Ice builds will take advantage of
either the Avalanche Key Passive or the Shatter Key Passive, depending on how much your build relies on crowd control\n\nFire builds are all about big
explosions and burning your enemies. For these builds, you’ll typically use either the Combustion Key Passive for burning enemies or the Esu’s Ferocity Key
Passive for chunky critical strikes.\n\nWhen making your own build for a Sorcerer, you’ll likely end up sticking to one element for your main damage sources.
But that’s not the most important thing you need to remember. Sorcerers die very, very easily in Diablo 4, and they have the weakest defensive stats by far —
however, they can deal a massive amount of damage when built correctly.'}, {'page': 'The best Sorcerer builds and skills in Diablo 4 (Polygon,
2023-10-18T21:43:17+00:00)', 'passage': 'The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)\nIf you don’t want to end up dead
more often than not, make sure to invest heavily in defensive skills while making your way toward one of the six Key Passives we listed above.\n\nLooking for
other Diablo 4 builds? We have explainers on the Barbarian, Rogue, Druid, and Necromancer classes.'}]
barbarianEvidence: [{'page': 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)', 'passage': 'The best Barbarian builds and
skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)\nBarbarians are a relatively simple class in Diablo 4: They smash stuff. However, Diablo 4 is
underscored by math that’s so complex that even the simple act of smashing stuff becomes a dizzying equation. That’s where we come in, with our suggestions of
the best Barbarian builds for Diablo 4 season 2.\n\nBarbarians are big, muscular, and powerful. They excel at using Bleed to kill their enemies over time. And
their unique weapon-swapping system means they get to have more Legendary powers at their disposal than any other class.\n\nIn this Diablo 4 guide, we’ve
gathered and simplified the best Diablo 4 Barbarian builds for season 2. If you need even more information on item affixes or other more complex ideas, we’ve
been sure to link out to the more detailed versions of these builds, as well as where to find their original creators.'}, {'page': 'The best Barbarian builds
and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)', 'passage': 'The best Barbarian builds and skills in Diablo 4 (Polygon,
2023-10-19T17:16:31+00:00)\nBest Barbarian leveling build\n\nThis build is based off Lexyu’s Barbarian leveling guide at IcyVeins, and is purely for leveling:
i.e., allocating your 58 skill points between levels 1 and 50. If you want more detailed information about the item affixes you want to look for on your gear,
or which other Legendary powers are great for your build, check out the original IcyVeins guide.\n\nBest skills for leveling\n\nMost Barbarian leveling guides
you’ll find make use of Hammer of the Ancients as your primary skill, as it deals massive damage while remaining very cheap to cast. In this build, you’ll also
use Frenzy, Death Blow, Rallying Cry, Leap, and Wrath of the Berserker.\n\nHere are the best skills for a Barbarian leveling build, organized by the exact order
you should purchase them in — bearing in mind that some skills require you to allocate a number of skill points in the tree in order to unlock them. As such,
you’ll need to unlock new skills before enhancing existing'}, {'page': 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)',
'passage': 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)\nAs such, you’ll need to unlock new skills before enhancing
existing ones:\n\nFrenzy / Enhanced Frenzy Hammer of the Ancients / Enhanced Hammer of the Ancients / Furious Hammer of the Ancients Combat Frenzy Rallying Cry
/ Enhanced Rallying Cry / Tactical Rallying Cry Hammer of the Ancients (ranks 2-3) Leap / Enhanced Leap / Power Leap Hammer of the Ancients (ranks 4-5) Death
Blow / Enhanced Death Blow / Warrior’s Death Blow / Death Blow (rank 2-5) Wrath of the Berserker / Prime Wrath of the Berserker / Supreme Wrath of the Berserker
Pit Fighter (ranks 1-3) Tempered Fury Furious Impulse (ranks 1-3) Walking Arsenal Hamstring No Mercy Expose Vulnerability (ranks 1-3) No Mercy (ranks 2-3)
Swiftness (ranks 1-3) Thick Skin (ranks 1-3) Counteroffensive (ranks 1-3) Defensive'}, {'page': 'The best Barbarian builds and skills in Diablo 4 (Polygon,
2023-10-19T17:16:31+00:00)', 'passage': 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)\n(ranks 1-3) Counteroffensive
(ranks 1-3) Defensive Stance (ranks 1-3) Slaying Strike (ranks 1-3) Aggressive Resistance Prolific Fury\n\nBest Weapon Expertise setup for leveling\n\nOnce you
unlock the Weapon Expertise system for your Barbarian, you’ll want to place the two-handed sword in your Technique slot. This will cause all of your Hammer of
the Ancients damage to make enemies Bleed.\n\nBest Legendary Aspects for Barbarian leveling build\n\nWhen you first start your new character, complete these
three dungeons first in order to collect their Aspects. Then place them on a piece of gear (ideally jewelry, when possible). Just keep in mind that you should
replace these Legendaries with new versions using the imprint system every few levels.\n\nAspect of Ancestral Force (Sunken Ruins dungeon in Scosglen) — Hammer
of the Ancients quakes outwards,'}, {'page': 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)', 'passage': 'The best
Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)\ndealing X% of its damage to enemies\n\n(Sunken Ruins dungeon in Scosglen) — Hammer
of the Ancients quakes outwards, dealing X% of its damage to enemies Aspect of the Expectant (Underroot dungeon in Scosglen) — Attacking enemies with a Basic
Skill increases the damage of your next Core Skill cast by X% up to X%\n\n(Underroot dungeon in Scosglen) — Attacking enemies with a Basic Skill increases the
damage of your next Core Skill cast by X% up to X% Edgemaster’s Aspect (Oldstones dungeon in Scosglen) — Skills deal X% increased damage based on Primary
Resource when cast, receiving the maximum benefit while you have full Primary Resource\n\nBarbarian Season 2 endgame build — Double Swing\n\nThis build is based
off of AaronActionRPG’s Double Swing Barbarian endgame guide at IcyVeins, and is a general endgame guide capable of finding success in all activities.'},
{'page': 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)', 'passage': 'The best Barbarian builds and skills in Diablo 4
(Polygon, 2023-10-19T17:16:31+00:00)\nand is a general endgame guide capable of finding success in all activities. This guide assumes that you’re over level 50,
have access to all 58 skill points, and are making progress on your Paragon Boards. If you want more detailed information about the item affixes you want to
look for on your gear, or why these Legendary powers are great for your build, check out the original IcyVeins guide.\n\nBest skill points for Double Swing
Barbarian endgame build\n\nThis build utilizes Double Swing as its core skill. It also uses Steel Grasp, Ground Stomp, Iron Maelstrom, Charge, and Lunging
Strike.\n\nSince you need all 58 points for this build, the order you select these skills in doesn’t matter as long as you have the pre-requisite points in the
earlier Skill Tree tiers. Here is how you should spend your Skill Points for this build:\n\nLunging Strike / Enhanced Lunging Strike / Combat Lunging
Strike\n\nDouble Swing (rank 1-5) / Enhanced Double Swing / Furious Double Swing\n\nEndless'}, {'page': 'The best Barbarian builds and skills in Diablo 4
(Polygon, 2023-10-19T17:16:31+00:00)', 'passage': 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)\nthis build, you’ll
want to take One-Handed Mace in your Technique.\n\nVampiric Powers for Double Swing Barbarian build\n\nSeason 2 adds the new Vampiric Powers, which allow you to
slot five additional abilities into your character to give you new effects. There are 22 total Vampiric Powers, and each are class agnostic, meaning they’re not
unique to any one class.\n\nAaronActionRPG recommends these five Vampiric Powers:\n\nAnticipation\n\nPrey on the
Weak\n\nRavenous\n\nDomination\n\nMetamorphosis\n\nTo insert those powers you’ll need to have these Pacts available on your armor:\n\n8 Ferocity\n\n3
Divinity\n\n2 Eternity\n\nLegendary Powers and Uniques for Double Swing Barbarian build\n\nTo make this build work, you’re going to need powerful Legendary
Powers and Unique items to augment your build.\n\nAaronActionRPG recommends these items:\n\nHelm : Weapon Master’s Aspect\n\n: Weapon Master’s Aspect Chest :
Aspect of Numbing Wrath (dungeon drop)\n\n: Aspect of'}, {'page': 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)',
'passage': 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)\nWeapon Master’s Aspect Chest : Aspect of Numbing Wrath
(dungeon drop)\n\n: Aspect of Numbing Wrath (dungeon drop) Gloves : Aspect of Anemia (dungeon drop)\n\n: Aspect of Anemia (dungeon drop) Pants : Aspect of Might
(dungeon drop)\n\n: Aspect of Might (dungeon drop) Boots : Exploiter’s Aspect\n\n: Exploiter’s Aspect Two-handed Bludgeoning : Aspect of Retribution (dungeon
drop)\n\n: Aspect of Retribution (dungeon drop) Two-handed Slashing : Accelerating Aspect\n\n: Accelerating Aspect Dual Wield 1 : Edgemaster’s (dungeon
drop)\n\n: Edgemaster’s (dungeon drop) Dual Wield 2 : Aspect of Berserk Ripping (dungeon drop)\n\n: Aspect of Berserk Ripping (dungeon drop) Amulet : Earthquake
Aspect\n\n: Earthquake Aspect Ring 1 : Aspect of Berserk Fury\n\n: Aspect of Berserk Fury Ring 2: Aspect of Limitless Rage\n\nIf you want to get into the nitty
gritty'}, {'page': 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)', 'passage': 'The best Barbarian builds and skills in
Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)\nRing 2: Aspect of Limitless Rage\n\nIf you want to get into the nitty gritty of stat priorities, refer to
AaronActionRPG’s original guide.\n\nParagon Boards for Double Swing Barbarian build\n\nThe Paragon Board system is very complex, and you’ll want to carefully
pair your Glyphs with specific boards for your class:\n\nStarter Board / Bloodfeeder Glyph Decimator / Ambidextrous Glyph Carnage / Might Glyph Flawless
Technique / Ire Glyph Weapon Master / Territorial Glyph Bone Breaker / Exploit Glyph Warbringer / Wrath Glyph\n\nYou should level your Glyphs to 15 in this
specific order:\n\nExploit Territorial Ambidextrous Might Wrath Bloodfeeder Ire\n\nYou can path your own way through the Paragon Boards following these basic
guidelines. However, for the best results, follow AaronActionRPG’s path exactly.\n\nHow to make a great Barbarian build\n\nBarbarians are Diablo 4’s only pure
melee class, with the other classes having hybrid builds for multiple ranges.'}, {'page': 'The best Barbarian builds and skills in Diablo 4 (Polygon,
2023-10-19T17:16:31+00:00)', 'passage': 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)\nThere are four main ideas that
all Barbarian builds stem from:\n\nFury dumping (most popular in Diablo 4’s pre-season)\n\nWeapon swapping\n\nBleed\n\nBerserk\n\nFury dump builds make use of
the Unbridled Rage Key Passive, which causes your Fury spenders to cost double the Fury but deal more than double the damage. These kinds of builds obviously
require very steady Fury generation, as it’s easy to starve yourself. Berserk builds that use the Unconstrained Key Passive are all about keeping your Barbarian
in the Berserk stage at all times, which nets you a hefty bonus. All Barbarian builds will likely use some skills that make them go Berserk, the Unconstrained
Key Passive just enhances the effect.\n\nBleed builds use the Gushing Wounds Key Passive to create a critical strike heavy build that’s great for whittling down
groups of enemies with damage over time and explosions. Weapon swapping builds offer what is almost certainly the most unique playstyle for Barbarians and takes
advantage of the Walking Arsenal Key Passive.'}, {'page': 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)', 'passage':
'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)\nThe idea here is that you’re constantly swapping between your skills and
weapons to give yourself massive damage buffs.\n\nWhen making a build, you want to have one of these Key Passives in mind and choose your skills accordingly.
Thankfully skills that share certain aspects (like Bleed) are color-coded on the skill tree, and you can even use the in-game search function to find them all.
Other Barbarian builds let you take advantage of mechanics like Thorns or Shouts. But they still typically end with one of these Key Passives.\n\nLooking for
other Diablo 4 builds? We have explainers on the Sorcerer, Rogue, Druid, and Necromancer classes.'}]
</item 8>

<item 9>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
refer to Northwar’s original guide.

Paragon Boards for Sorcerer Firewall build

The Paragon Board system is very complex, and you’ll want to carefully pair your Glyphs with specific boards for your class:

Starter Board / Reinforced Glyph Burning Instinct / Destruction Glyph Searing Heat / Adept Glyph Static Surge / Control Glyph Enchantment Master / Exploit

You should level your Glyphs to 15 in this specific order:

Destruction Control Adept Exploit Reinforced

You can path your own way through the Paragon Boards following these basic guidelines. However, for the best results, follow Northwar’s path exactly.

How to make a great Sorcerer build

Sorcerers are Diablo 4’s pure caster class, and use elemental magic to differentiate from the spells of the Druid and Necromancer. The Sorcerer has six Key
Passives, but only three elements, which is what you’ll base your build around:

Lightning

Fire

Ice

Lightning builds utilize Lightning skills to deal with large groups of enemies at once.

Graph facts:
- The best Sorcerer builds and skills in Diablo 4 -MENTIONS-> Enchantment
- 10 scorching-hot discoveries made about the sun in 2023 -MENTIONS-> lightning
- 54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods -MENTIONS-> lightning
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> lightning
- The best Sorcerer builds and skills in Diablo 4 -MENTIONS-> Sanctuary

Entities: lightning, ice, fire, The best Sorcerer builds and skills in Diablo 4, Sorcerer Firewall build, Northwar, Paragon Board system, Glyphs, Starter Board,
Reinforced Glyph, Burning Instinct, Destruction Glyph, Searing Heat, Adept Glyph, Static Surge
</item 9>

<item 10>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Recharging Aspect (Zenith in Fractured Peaks) — Each time Chain Lightning bounces off you, gain X Mana

(Zenith in Fractured Peaks) — Each time Chain Lightning bounces off you, gain X Mana Aspect of Control (Sunken Library in Kehjistan) — You deal X% more damage
to Immobilized, Stunned, or Frozen enemies

(Sunken Library in Kehjistan) — You deal X% more damage to Immobilized, Stunned, or Frozen enemies Edgemaster’s Aspect (Oldstones dungeon in Scosglen) — Skills
deal X% increased damage based on Primary Resource when cast, receiving the maximum benefit while you have full Primary Resource

Sorcerer Season 2 endgame build — Firewall

This build is based off of Northwar’s Sorcerer endgame guide at MaxRoll.gg, and is a general endgame guide capable of finding success in all activities.
</item 10>
=== END ITEM id=785b7b12c0d1cfc0 ===

=== ITEM id=7434bf9065ceea26 ===
Gold sentences and facts:
Sentence 0: Wired removed the article for not meeting its standards, but the degree to which it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part.
  Fact 0.0: Wired removed the article for not meeting its standards.
  Fact 0.1: The article was shared widely on social media.
  Fact 0.2: The article was boosted in write-ups at other outlets.
  Fact 0.3: The degree to which the article was shared and boosted shows how willing people are to accept foul play on Google's part.
Sentence 1: When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.
  Fact 1.0: Google's search head Prabhakar Raghavan testified in court on October 28.
  Fact 1.1: Prabhakar Raghavan revealed that Google paid $26.3 billion in 2021 to multiple browsers, phones and platforms.
  Fact 1.2: The payments went to companies including Apple, Samsung and Mozilla.
  Fact 1.3: The Verge reports this.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google siphons off news publishers' content, their readers and ad revenue through anticompetitive means.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
“When a user searches for information on a topic, Google displays a ‘Knowledge Panel’ to the right of the search results. This panel contains a summary of
content drawn from the Knowledge Graph database,” the complaint states. “Google compiled this massive database by extracting information from Publishers’
websites — what Google calls ‘materials shared across the web’ —and from ‘open source and licensed databases,'” it says.

By 2020, the Knowledge Graph had grown to 500 billion facts about 5 billion entities. But much of the “collective intelligence” that Google tapped into was
content “misappropriated from Publishers,” the complaint alleges.

Other Google technologies, like “Featured Snippets” where Google algorithmically extracts answers from webpages, were also cited as shifting traffic away from
publishers’ websites.

More importantly, perhaps, is the suit’s tackling of how AI will impact publishers’ businesses. The problem was recently detailed in a report on Thursday by The
Wall Street Journal, which led with a shocking statistic.

Graph facts:
- The inside story of Dave Clark's tumultuous last days at Flexport -MENTIONS-> The Wall Street Journal
- Europe’s artists and labels could lose $137m annually to US recorded music rightsholders if EU law isn’t changed, trade groups warn -MENTIONS-> Publishers
- A comprehensive list of 2023 tech layoffs -MENTIONS-> The Wall Street Journal
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> The Wall Street Journal
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet

Entities: Knowledge Graph, Publishers, Knowledge Panel, Featured Snippets, class action antitrust suit, misappropriation of Publishers’ content, The Wall Street
Journal, report on Thursday, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line, antitrust, TechCrunch, AI,
Google
</item 1>

<item 2>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google Search
- The people who ruined the internet -MENTIONS-> Google Search
- Early impressions of Google’s Gemini aren’t great -MENTIONS-> Google Search
- The restaurant nearest Google -MENTIONS-> Google Search
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Google Search

Entities: Google Search, Is Google Search better than the rest? And is that fair?, Grey, DuckDuckGo, privacy invasion, inappropriate data handling, organic
search results, ad systems affecting organic results, Danny Sullivan, X, The Age, Google, WIRED, ad revenue
</item 2>

<item 3>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Alphabet
- Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more -MENTIONS-> Alphabet
- ASX set to rise ahead of inflation report after big tech lifts Wall Street -MENTIONS-> Alphabet
- Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation -MENTIONS-> Alphabet
- Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas -MENTIONS-> Alphabet

Entities: Alphabet, Helena World Chronicle, U.S. District Court in D.C., Search Generative Experience, Bard AI chatbot, Knowledge Graph, anticompetitive
behavior, U.S. antitrust law, Sherman Act, Arkansas, question-and-answer technologies, News publisher files class action antitrust suit against Google, citing
AI’s harms to their bottom line, class action lawsuit, TechCrunch, AI technologies
</item 3>

<item 4>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Helena World Chronicle
- Alphabet -ACCUSED_OF-> Sherman Act
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Sherman Act
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. District Court in D.C.

Entities: Sherman Act, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line, U.S. Justice Department,
Hausfeld, Helena World Chronicle, LLC, class action antitrust suit against Google, U.S. Justice Department lawsuit against Google, 2020 Justice Department civil
antitrust suit over search and search advertising, digital ad technologies, Clayton Act, search and search advertising, digital news and reference publishing,
class-wide monetary and injunctive relief, free marketplace of ideas, free press
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Helena World Chronicle
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. District Court in D.C.
- Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
-MENTIONS-> Washington Post
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Search Generative Experience

Entities: SGE, Google Bard AI, GoogleBot, Google general search service, AdSense, News Media Alliance, Washington Post, Epic Games lawsuit against Google over
app store antitrust issues, app store antitrust issues, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line,
class action antitrust suit against Google, Epic Games, Allen Institute for AI, TechCrunch, Google
</item 5>

<item 6>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 6>

<item 7>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Bing
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Bing
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> monopoly power
- Is Google Search better than the rest? And is that fair? -MENTIONS-> Bing

Entities: monopoly power, FTC-Facebook case, Bing, Apple, WhatsApp, TechCrunch, data, Google antitrust case, 5 things we learned so far about the Google
antitrust case, DOJ, FTC, FTC lawsuit against Amazon, anticompetitive and unfair strategies, DOJ investigation of Apple, Apple's policy for third-party apps on
its devices
</item 7>

<item 8>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 8>

<item 9>
Graph query row:
article: Is Google Search better than the rest? And is that fair?
published_on: 2023-10-22T18:00:00+00:00
publisher: The Age
Google: Google
alleged_practices: ['US v Google trial', 'anticompetitive behavior', 'U.S. antitrust law', 'Sherman Act', 'class action antitrust suit', 'misappropriation of
Publishers’ content', 'Epic Games lawsuit against Google over app store antitrust issues', 'class action antitrust suit against Google', 'digital ad
technologies', 'anti-competitive practices', 'search monopoly', 'antitrust', 'monopoly', 'anti-competition law', 'systematically destroying chats',
'anticompetitive conduct', 'antitrust law', 'unlawful monopoly', 'making potential evidence disappear', 'illegal tying', 'Android app distribution', 'Android
in-app payment processing', 'deleting potentially incriminating messages']
article_mentions: ['Chrome', 'ads', 'Mozilla', 'Firefox', 'DuckDuckGo', 'Google Search', 'Kent Walker', 'trial on Google', 'technology industry', 'Yahoo!',
'Wired opinion article', 'Grey', 'privacy invasion', 'inappropriate data handling', 'organic search results', 'ad systems affecting organic results', 'organic
results', 'keyword matching', 'Web & App Activity', 'search quality raters', 'quality tests', 'Megan Grey', 'semantic keyword matching', 'Bing Chat', 'Bing',
'Microsoft', 'Danny Sullivan', 'Google', 'Walker', 'X', 'US', 'lawsuit', 'WIRED', 'ad revenue', 'machine learning', 'Windows', 'generative AI']
</item 9>

<item 10>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> The Wall Street Journal
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> The Wall Street Journal
- report on Thursday -PUBLISHED_BY-> The Wall Street Journal
- Publishers -PARTICIPATED_IN-> class action antitrust suit
- The inside story of Dave Clark's tumultuous last days at Flexport -MENTIONS-> The Wall Street Journal

Entities: class action antitrust suit, The Wall Street Journal, deal to license news for AI model training, News publisher files class action antitrust suit
against Google, citing AI’s harms to their bottom line, antitrust, The Atlantic, Axel Springer, TechCrunch, AI, Similarweb, Google, OpenAI
</item 10>
=== END ITEM id=7434bf9065ceea26 ===
