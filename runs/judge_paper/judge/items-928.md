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

This file has 4 items.

=== ITEM id=16f4fb0452cc9986 ===
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
Diablo 4 season 2’s biggest draw is its quality-of-life updates (Polygon, 2023-10-04T20:01:00+00:00)
All of these changes, as well as the content for Season of Blood, will come to the game on Oct. 17, just a few weeks before fans are expecting Blizzard to
announce Diablo 4’s first expansion at BlizzCon 2023.
</item 1>

<item 2>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
and is a general endgame guide capable of finding success in all activities. This guide assumes that you’re over level 50, have access to all 58 skill points,
and are making progress on your Paragon Boards. If you want more detailed information about the item affixes you want to look for on your gear, or why these
Legendary powers are great for your build, check out the original IcyVeins guide.

Best skill points for Double Swing Barbarian endgame build

This build utilizes Double Swing as its core skill. It also uses Steel Grasp, Ground Stomp, Iron Maelstrom, Charge, and Lunging Strike.

Since you need all 58 points for this build, the order you select these skills in doesn’t matter as long as you have the pre-requisite points in the earlier
Skill Tree tiers. Here is how you should spend your Skill Points for this build:

Lunging Strike / Enhanced Lunging Strike / Combat Lunging Strike

Double Swing (rank 1-5) / Enhanced Double Swing / Furious Double Swing

Endless
</item 2>

<item 3>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
Weapon Master’s Aspect Chest : Aspect of Numbing Wrath (dungeon drop)

: Aspect of Numbing Wrath (dungeon drop) Gloves : Aspect of Anemia (dungeon drop)

: Aspect of Anemia (dungeon drop) Pants : Aspect of Might (dungeon drop)

: Aspect of Might (dungeon drop) Boots : Exploiter’s Aspect

: Exploiter’s Aspect Two-handed Bludgeoning : Aspect of Retribution (dungeon drop)

: Aspect of Retribution (dungeon drop) Two-handed Slashing : Accelerating Aspect

: Accelerating Aspect Dual Wield 1 : Edgemaster’s (dungeon drop)

: Edgemaster’s (dungeon drop) Dual Wield 2 : Aspect of Berserk Ripping (dungeon drop)

: Aspect of Berserk Ripping (dungeon drop) Amulet : Earthquake Aspect

: Earthquake Aspect Ring 1 : Aspect of Berserk Fury

: Aspect of Berserk Fury Ring 2: Aspect of Limitless Rage

If you want to get into the nitty gritty
</item 3>

<item 4>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
It’s the first heavy-duty chamber LARP I’ve played since the pandemic; it reset my brain and helped me remember what I value in play. Players continuously swap
roles as they play through a very queer Greek myth in which young Orestes returns home and must decide whether to kill his treacherous mother and father-in-law.
Never have I planned a murder with such erotically charged practice stabbing.

Dungeon Crawl Classics

Thilo Graf of AAW Games

I run and play a ton of different RPG systems, but precious few have had such a tremendous and lasting impact on my design and games as Dungeon Crawl Classics
by Goodman Games.

With aesthetics influenced by all-time greats like Leiber and Moorcock, volatile magic, a relatively flat power curve for a d20-based game, and an emphasis on
the skill of the players, as opposed to primarily rewarding the crafting of effective builds, DCC is a joy to run and play.
</item 4>

<item 5>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
They also get a way to heal themselves and way to get a second attack once per short rest.

At level 3, you get to choose a subclass from Battle Master, Eldritch Knight, and Champion. Battlemasters get a set of Superiority Dice that let you perform
Maneuvers to control the flow of battle and the battlefield. Eldritch Knights add magic into the Fighter’s arsenal. Champions keep it simple with a better
change to deal a Critical Hit (for double damage).

Champion is the best subclass for a Fighter. Consult our Fighter build guide for more details on why, and to see which skills and feats to pick.

Best ranged class for beginners: Ranger

Rangers are agile hunters that focus on Dexterity as their primary Ability. Rangers are an interesting mix of melee, stealth, skills, and magic. That also makes
it harder to get the most out of the class. Rangers get Light and Medium Armors and some of the more basic weapons.
</item 5>

<item 6>
Fortnite is the latest game to entice players with a portal to the past (TechCrunch, 2023-10-30T19:18:39+00:00)
However, the WoW Classic team’s philosophy has evolved since it was released: [Blizzard] initially was very serious about releasing the game without any
changes, but these days the team feels that some changes need to be made,” Harper said.

“I’m not an expert in Fortnite, but a smart game developer will pay close attention to the features that spark that nostalgic joy and keep those in the
forefront, while considering retaining modern conveniences that players appreciate.”

Epic hasn’t shown its full hand for the new-old version of its own game, but it does sound like some of Fortnite’s quality of life improvements will be sticking
around, including sprinting and “mantling,” a parkour-like move that lets players climb over obstacles. In recent years, Epic has sprinkled all sorts of zany
mobility options onto its map, letting players fall through the sky, pilot biplanes or (our personal preference) roll in a giant hamster ball to get around
faster.

“With MMOs, it’s easier than ever for gaming experiences to be lost completely to time.
</item 6>

<item 7>
Diablo 4 season 2’s biggest draw is its quality-of-life updates (Polygon, 2023-10-04T20:01:00+00:00)
Shely told the panel that last season, players set a goal for themselves: Get to level 100. However, many players found that they were never able to hit that
milestone because XP gains were so slow. This change should allow more players to reach the level cap each season, allowing them to play with a truly maxed-out
build.

As for how players will spend their time at max level and where they’ll be able to earn some of the game’s most powerful items, enter the five new endgame
bosses: Grigoire The Galvanic Saint, the Echo of Varshan, The Beast in the Ice, the Dark Master, and the Echo of Duriel. Each boss has its own laundry list of
busy work for players to complete in order to unlock their encounter — such as completing Nightmare Dungeons above level 30 or completing Grim Favors — but they
also come with Unique rewards.
</item 7>

<item 8>
Diablo 4 season 2’s biggest draw is its quality-of-life updates (Polygon, 2023-10-04T20:01:00+00:00)
That’s Unique with a capital U.

Because the bosses each offer their own pool of Unique items, this process makes farming for that special helmet that much easier. Instead of running dungeons
and hoping the desired item drops, players can just gather all the materials, kill The Beast in the Ice, and then do it all again if they don’t get what they
wanted. The Echo of Duriel is perhaps the most interesting of the five, as Duriel has a chance to drop the ultimate, super-rare, uber Uniques that were
previously almost impossible to acquire. To compensate, players will need to kill the four other endgame bosses each time they want to face Duriel.

Blizzard’s designers didn’t build these bosses to be as tough as the ultimate Uber Lilith fight. However, these bosses will (hopefully) prove a challenge to
most players while still giving them a targeted way to farm some must-have items.
</item 8>

<item 9>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Recharging Aspect (Zenith in Fractured Peaks) — Each time Chain Lightning bounces off you, gain X Mana

(Zenith in Fractured Peaks) — Each time Chain Lightning bounces off you, gain X Mana Aspect of Control (Sunken Library in Kehjistan) — You deal X% more damage
to Immobilized, Stunned, or Frozen enemies

(Sunken Library in Kehjistan) — You deal X% more damage to Immobilized, Stunned, or Frozen enemies Edgemaster’s Aspect (Oldstones dungeon in Scosglen) — Skills
deal X% increased damage based on Primary Resource when cast, receiving the maximum benefit while you have full Primary Resource

Sorcerer Season 2 endgame build — Firewall

This build is based off of Northwar’s Sorcerer endgame guide at MaxRoll.gg, and is a general endgame guide capable of finding success in all activities.
</item 9>

<item 10>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Your Charisma Increases by 1, to a maximum of 20. Your Proficiency Bonus is also doubled for Deception and Performance Checks.

. Your Charisma Increases by 1, to a maximum of 20. Your Proficiency Bonus is also doubled for Deception and Performance Checks. Elemental Adept . Your spells
ignore Resistance to a damage type of your choice. When you cast spells of that type, you cannot roll a 1.

. Your spells ignore Resistance to a damage type of your choice. When you cast spells of that type, you cannot roll a 1. Moderately Armored . You gain Armor
Proficiency with Medium Armor and shields, and your Strength or Dexterity increases by 1, to a maximum of 20.

. You gain Armor Proficiency with Medium Armor and shields, and your Strength or Dexterity increases by 1, to a maximum of 20. Spell Sniper.
</item 10>

<item 11>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
It is not as much of a heroic power fantasy as D&D 5e or Pathfinder, but neither is it as minimalist as many old-school games. With fickle gods, mutating
magics, lucky scoundrels, and ferocious fighters, DCC feels like playing through a Bal-Sagoth or Eternal Champion song. It’s also easy to pick up if your group
has any experience with D&D’s third edition or Pathfinder. DCC rocks! Its focus on simple, impactful rules was a core design inspiration for the (almost
system-neutral) Survivalist’s Guide to Spelunking for 5e, a book I wrote with Douglas Niles and Stephen Yeardley. It warms my heart to know that DCC authors,
judges, and players use the book as well!

Dungeons & Dragons

Mike Haracz, corporate R&D chef and host of Heroes’ Feast

I’m lucky enough to be in the seventh year of a homebrew Dungeons & Dragons campaign, one I play with friends locally in the Chicagoland area.
</item 11>

<item 12>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
It is up to you to go on a perilous quest of your own to retrieve your master’s fabled weapon in time for their final duel.

Throughout this transformational quest, you use three six-sided dice to resolve mechanics for racing against time, overcoming treacherous encounters, and
uncovering new truths about your master’s shrouded past. Embrace your inner doggy, buy this 17-page game, and begin your adventure today! Swiftly now — your
master’s fate is in your paws!

Home

Wesley “Nox” Crowe, aka Noxweiler Berf, creative director for Hunters Entertainment

Doug Levandowski and Yeonsoo Julian Kim’s Home drew me in immediately, and has been one of my most recommended tabletop role-playing games. If you know me or my
work, then the fact that it’s a horror game probably won’t come as much of a surprise. It’s a haunted house exploration simulator that can be played with a full
table of players or alone as a solo journaling experience.
</item 12>

<item 13>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
They’re the most magic-focused of the melee classes. They get all armors and shields, along with a good mix of weapons. Choosing from the three subclasses —
Oath of the Ancients, Oath of Devotion, and Oath of Vengeance — determines what kind of magic you get. Oath of the Ancients adds in healing magic, Oath of
Devotion gets defense-focused magic, and Oath of Vengeance adds offensive magic. Paladins also get a handful of other spells for dealing out even more holy
justice.

Oath of the Ancients is the best subclass for a Paladin. Consult our Paladin build guide for more details on why, and to see which skills and feats to pick.

Best class for speed: Monk

Monks are a melee class kind of like Fighters, but with a lot more pizzazz. Their best Ability is Dexterity, so their combat is more martial arts than
swordplay.
</item 13>

<item 14>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
It’s a time in your life where you feel so, so much. You can see that earnestness in the teenage experience where everything is a big, huge issue — sometimes to
the point of cringe — tied up in that big, global issue of the meteor that’s looking to destroy everything. —N. Carpenter

Related August games you might have missed

40. World of Warcraft Classic

Developer: Blizzard Entertainment

Where to play: Mac and Windows PC

Something very, very interesting started to happen in the retro version of WoW this year. While Blizzard has been content to march one half of the game’s
community forward through its history of expansions, it’s started to come up with creative ways to keep the other half — the half that wants to stay in the game
as it was at launch — engaged.

The first of these was Hardcore, a brilliant permadeath mode that instantly made this aging game both more dangerous and more social, resurrecting the spirit of
its 2004 servers.
</item 14>

<item 15>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
It uses custom tarot cards and a whiteboard to track your character’s progress as they dare to shine a light on the chilling danger of a variety of potential
scenarios full of ghosts and other dark entities.

​The action is propelled through the use of the custom tarot cards — the Night Deck — which are filled with prompts that require the creative interaction of
players. Each round, players take turn as the director of the story and guide the narrative forward using the guidance of the cards to describe and draw each
room of the haunted house that they are exploring. The game encourages collaboration in determining the story’s direction and does a remarkable job of steering
the action forward without a single game master putting forth the often large amount of preparation that role-playing games usually require.

​As the game progresses, characters collect wounds and push past the terrors that await them and work together to resolve the mysteries at the root of each
horror. The solo mode of the game plays just as smoothly as the group experience, leaning on the prompts and player’s imagination to create a journal
documenting the character’s harrowing encounter.
</item 15>

<item 16>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
The book itself has a stunning and distinct art style that immediately transports you into the doom metal fantasy of the setting; but that’s only a small part
of its allure.

Unlike many TTRPGs that one merely has aspirations of playing one day, Mörk Borg does everything in its power to make the game easy to run. Although it’s a
short read, no line is wasted. Even item descriptions are dripping with hints at nefarious deeds one must have done to acquire them. The optional classes are an
absolute delight for those of us that savor the idea of playing the unsavory. From the esoteric introverts that choose to make their hovel far from prying eyes
to the unfortunate discarded-at-birth cutthroats who’ve had to scratch out a living any way they can, there is something for everyone. There is always the
option of creating something more tailor-made if none of the classes are appealing, but I appreciate that Mörk Borg offers beginner-friendly ideas and
archetypes to help you create something that fits right into the gritty setting.
</item 16>

<item 17>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
It was an ambitious promise, to be sure, but four years and a whole pandemic after the studio announced the long-awaited sequel at BlizzCon 2019, it’s here, and
it’s fantastic.

But Diablo 4’s marriage of tone, action, and role-playing isn’t what makes it so good. In addition to all of those other things, Diablo 4 is the best launch
we’ve seen for a new “living game” in recent memory.

With the likes of Destiny, Anthem, and even Diablo 3, it was clear from the start that there were some nuggets of potential. But being a fan meant slogging
through mountains of frustration just to taste a morsel of what you’d hope those games would become. Playing these games early on was a kind of gamble.

Diablo 4, however, is unlike any of those projects, because its systems were deep and nuanced from the start, enough to spend hundreds of hours growing your
character. And there is already loads of content to support that kind of time investment.
</item 17>

<item 18>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
Barbarians are a relatively simple class in Diablo 4: They smash stuff. However, Diablo 4 is underscored by math that’s so complex that even the simple act of
smashing stuff becomes a dizzying equation. That’s where we come in, with our suggestions of the best Barbarian builds for Diablo 4 season 2.

Barbarians are big, muscular, and powerful. They excel at using Bleed to kill their enemies over time. And their unique weapon-swapping system means they get to
have more Legendary powers at their disposal than any other class.

In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Barbarian builds for season 2. If you need even more information on item affixes or
other more complex ideas, we’ve been sure to link out to the more detailed versions of these builds, as well as where to find their original creators.
</item 18>

<item 19>
Diablo 4 season 2’s biggest draw is its quality-of-life updates (Polygon, 2023-10-04T20:01:00+00:00)
Previously, players would have to max out their Renown — which all but required doing nearly every dungeon and most side quests in the game — every season in
order to get all available Skill and Paragon Points. Not only is that not the case anymore, but (unlike with the Altars of Lilith carryover) you won’t need to
first log into a character that’s completed the Renown track in order to propagate that data to the rest of your account. Piepiora told us that all of that data
migration should happen without that extra step in season 2.

This is yet another massive 180 on the Renown system from Blizzard in barely five months’ time. When we asked about the massive shift, Piepiora told us that
players simply weren’t having fun redoing the grind every season, hence the move here (as with the Altars last season) to alleviate the pressure on players to
only play “efficiently.”
</item 19>

<item 20>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Where Bards come into their own, though, is with Bardic Inspiration — a way to boost the rolls of their allies — and their spells that are a nice mix of utility
and offensive magic.

Bards get three subclasses at level 3: College of Lore, College of Valor, and College of Swords. College of Lore Bards get a lot of useful skills to use outside
of combat and get to use their Bardic Inspiration to make enemies weaker, College of Valor Bards help their allies hit even harder, and College of Swords Bards
focus on a swashbuckling style of combat.

College of Valor is the best subclass for a Bard. Consult our Bard build guide for more details on why, and to see which skills and feats to pick.

Best class for magic damage: Wizard

Wizards are the archetypical caster class. They are the only Intelligence-based class in Baldur’s Gate 3 and have the most spells — and the most spell slots to
cast them — of all of the casters.
</item 20>

<item 21>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
I’ve rolled for every character, grinded out all of said characters, 100%-ed the exploration, completed every quest — did all the stuff that would burn anyone
out. Honkai: Star Rail is Hoyoverse’s answer to the Genshin Impact burnout.

The slower-paced turn-based combat, the ability to auto-battle (if your teams are strong enough), and the smaller maps make it the perfect daily game for me to
funnel my interest into without feeling exhausted. The characters are fun, flashy, and easy to latch onto. Should I main the sneaky but elegant Kafka, who uses
damage-over-time skills to whittle her opponents away? Or should I use the brooding Blade, who unleashes huge attacks at the cost of his HP? Ah, I guess I’ll
let my wallet decide — it is a gacha game, after all. —JL

20.
</item 21>

<item 22>
Diablo 4 season 2’s biggest draw is its quality-of-life updates (Polygon, 2023-10-04T20:01:00+00:00)
Diablo 4’s Season of Blood brings some cool new vampire powers into the fold, but the real star of the action game’s second season is its sweeping
quality-of-life changes. Players will now have an easier time leveling up, gaining quality loot, and just navigating the open world of Sanctuary.

Game director Joe Shely and associate game director Joe Piepiora sat down to do a roundtable interview with Polygon and other outlets ahead of Blizzard’s Oct.
4th livestream, and gave additional context to Nightmare Dungeon changes, Renown, and more.

One of the most notable changes for players in Diablo 4’s second season update will be an increase in experience gains. Blizzard claims players will be able to
reach the level cap of 100 about 40% faster than in Season of the Malignant — which already increased experience gains compared to Diablo 4’s launch. This
process should be much smoother for players as well, as monsters will now match their levels between 55 and 75.
</item 22>

<item 23>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
You can even wear them as mourning jewelry.

This Pathfinder 2nd edition-compatible setting has been as enchantingly addictive as the fictional ichor that propelled Ourichor into the renaissance era that
the project is named after. While there are only two exquisitely illustrated supplements on creatures and equipment available currently, I recommend those with
a love of the unusual to keep an eye out for more in the years to come, such as the upcoming Paragons of Luminant Age that introduces NPCs and explores their
motives through the four moons.

Marvel Multiverse Role-playing Game

As a role-player, I’ve spent a lot of time looking for a good lightweight superhero RPG. I’ve tried quite a few but had yet to find one that really hit the mark
for me until I tried Marvel Multiverse Role-playing Game. I have to say, I was initially dubious about this game but it has blown me out of the water.
</item 23>

<item 24>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Rangers get Light and Medium Armors and some of the more basic weapons.

Rangers also make a lot of choices during character creation (and leveling up) that determine how they play. At level 1, you’ll choose a Favored Enemy — a
preferred quarry — and a preferred environment that grants you some magical spells.

At level 3, you get to choose a Ranger subclass from Beast Master, Gloom Stalker, or Hunter. Beast Masters get a Companion animal that fights along side you,
Gloom Stalkers focus on stealth, and Hunters get to specialize their melee abilities.

Hunter is the best subclass for a Ranger. Consult our Ranger build guide for more details on why, and to see which skills and feats to pick.

Best class for damage: Barbarian

Barbarians are all about combat. They get light and medium armor (and shields), but they’ve also got the option to keep their AC up even without wearing armor.
</item 24>

<item 25>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
As such, you’ll need to unlock new skills before enhancing existing ones:

Frenzy / Enhanced Frenzy Hammer of the Ancients / Enhanced Hammer of the Ancients / Furious Hammer of the Ancients Combat Frenzy Rallying Cry / Enhanced
Rallying Cry / Tactical Rallying Cry Hammer of the Ancients (ranks 2-3) Leap / Enhanced Leap / Power Leap Hammer of the Ancients (ranks 4-5) Death Blow /
Enhanced Death Blow / Warrior’s Death Blow / Death Blow (rank 2-5) Wrath of the Berserker / Prime Wrath of the Berserker / Supreme Wrath of the Berserker Pit
Fighter (ranks 1-3) Tempered Fury Furious Impulse (ranks 1-3) Walking Arsenal Hamstring No Mercy Expose Vulnerability (ranks 1-3) No Mercy (ranks 2-3) Swiftness
(ranks 1-3) Thick Skin (ranks 1-3) Counteroffensive (ranks 1-3) Defensive
</item 25>

<item 26>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
As such, you’ll need to unlock new skills before enhancing existing ones:

Frost Bolt / Enhanced Frost Bolt Chain Lightning Glinting Frost Bolt Enhanced Chain Lightning / Greater Chain Lightning Teleport Frost Nova / Enhanced Frost
Nova / Mystical Frost Nova Chain Lightning (rank 2-5) Fireball (rank 1 only for the Enchantment slot) Lightning Spear Glass Cannon (rank 1-3) Devastation
Elemental Dominance (rank 1-3) Coursing Currents (rank 1-3) Elemental Attunement Ice Armor / Enhanced Ice Armor / Shimmering Ice Armor Enhanced Teleport /
Shimmering Teleport Enhanced Lightning Spear Vyr’s Mastery Invoked Lightning Spear / Lightning Spear (rank 2-5) Frost Bolt (rank 2-5) Conduction (rank 1-3)
Electrocution (rank 1-3) Align the Elements Protection (rank 1-3) Icy Veil (rank 1-3) Frost Nova (rank 2-3)

Best Enchantment setup for leveling

The Enchantment
</item 26>

<item 27>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
Swing (rank 1-5) / Enhanced Double Swing / Furious Double Swing

Endless Fury

Ground Stomp (rank 1-5) / Enhanced Ground Stomp / Strategic Ground Stomp

Charge (rank 1-5) / Enhanced Charge / Power Charge

Aggressive Resistance (rank 1-3)

Prolific Fury (rank 1-3)

Steel Grasp / Enhanced Steel Grasp / Fighter’s Steel Grasp

Thick Skin

Defensive Stance (rank 1-3)

Counteroffensive (rank 1-3)

Pit Fighter (rank 1-3)

No Mercy (rank 1-3)

Iron Maelstrom / Prime Iron Maelstrom / Supreme Iron Maelstrom

Duelist (rank 1-3)

Wallop

Concussion (rank 1-3)

Walking Arsenal

Arsenal System for the Double Swing Barbarian build

For this build, you’ll want to take One-Handed Mace in your Technique.
</item 27>

<item 28>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Granted, those upgrades are part of a monetization setup that gets a little heavy-handed, but if you’re patient, there’s plenty of game here for free. And
unlike in some of Niantic’s other games, the real-world elements don’t feel tacked on, with a design that blends almost perfectly with the company’s map tech,
allowing you to feel like you’re tracking monsters as you get outside and walk around. It all makes for a big step forward in fusing Niantic’s real-world tech
approach with combat and exploration mechanics that can stand up on their own. —Matt Leone

Related Monster Hunter Now is the most traditional Niantic game yet

28. Diablo 4

Developer: Blizzard Entertainment

Where to play: PlayStation 4, PlayStation 5, Windows PC, Xbox One, and Xbox Series X

With Diablo 4, Blizzard Entertainment set out to marry the frenetic action of Diablo 3, the deep RPG systems of Diablo 2, and the dark tone of the original
game.
</item 28>

<item 29>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
If you’re bound to The Archfey, you’ll gain an action named “Fey Presence,” which will charm or frighten nearby foes.

The Archfey also has two exclusive spells — Faerie Fire and Sleep.

Best Warlock subclass and build in BG3

Baldur’s Gate 3 races don’t really have any direct impact on your class, but certain races and subraces come with features that play well with certain classes
like extra movement or proficiency with certain weapons.
</item 29>

<item 30>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
For Warlocks, it’s best to focus on weapon and armor proficiencies:

Drow ’s Drow Weapon Training adds weapons like the rapiers and hand crossbows

’s Drow Weapon Training adds weapons like the rapiers and hand crossbows Humans ’ and Half-Elves ’ Civil Militia add spears, pikes, halberds, and glaives along
with shields

’ and ’ Civil Militia add spears, pikes, halberds, and glaives along with shields Githyanki Martial Prodigy means you get medium armor along with longswords and
greatswords

Martial Prodigy means you get medium armor along with longswords and greatswords Dwarven Combat Training gets you battleaxes, handaxes, light hammers, and
warhammers. The Shield Dwarf subrace also gets you Medium Armor

When you’re creating and leveling up your Warlock, Charisma (for spellcasting) is your most important stat.
</item 30>

<item 31>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
I love the characters, the environments, the writing, the lore, and the flow of battles. But I have concerns with the game’s pacing. By the end of the first
chapter, I had leveled up 16 times. Each level offered marginal rewards, like being able to move slightly farther during the character’s turn in combat or
having a higher parry chance when being attacked. A slow drip-feed system means each level feels less important, and even though I’m growing stronger, I don’t
get that sense of long-term satisfaction.

It’s especially frustrating to hit a roadblock like the one at the end of Act 1, where an incredibly tough boss rolls out of nowhere and spanks my crew — and I
can’t leave to go grind experience somewhere else. I eventually found out a way to cheese the fight by focusing on my melee fighters’ positioning, but it took
far too long banging my head against the wall. The victory tasted like ash in my mouth after all that frustration.
</item 31>

<item 32>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
They only get Light Armor, so they have to focus on staying unseen and quick. Their weapons tend to be simple and focus on ones that have the Finesse type —
meaning you can add your Dexterity modifier instead of your Strength. All Rogues get to use Dash (double movement), Disengage (don’t provoke Opportunity
Attacks), and Hide as Bonus Actions. They also get Sneak Attack combat actions that let them deal extra damage whenever they have Advantage.

Level 3 Rogues choose from the Thief, Arcane Trickster, and Assassin subclasses. Thief Rogues get an extra bonus action, Arcane Tricksters mix in magic, and
Assassins get advantages over surprised or unprepared enemies.

Thief is the best subclass for a Rogue. Consult our Rogue build guide for more details on why, and to see which skills and feats to pick.

Best class for melee and magic: Paladin

Paladins start out mixing in (Divine) magic with their melee abilities. They’re the most magic-focused of the melee classes.
</item 32>

<item 33>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
You play as a broody Clive Rosfield, a young man whose life’s work is to protect his little brother, Joshua. The story begins when Clive’s life takes a turn for
the worse and he vows to destroy the monster who ruined his and his family’s legacy.

Developed by Creative Business Unit III, Square Enix’s internal team behind the MMORPG Final Fantasy 14, 16 leans into patchwork territories of fantasy genre
fare. There is palace intrigue, a whole lot of sex, and endless war between nations. But the developers then sprinkle in Final Fantasy elements like mother
crystals, dazzling kaiju fights between summons (known as Eikons in this iteration), and of course, Chocobos.

The quality of the story in this long and linear character-driven RPG waxes and wanes, but the action combat is among the best I’ve ever played. The gameplay
grips you from the very beginning as Clive smoothly dashes, parries, and swings his giant sword and varied magic with a dazzling amount of style.
</item 33>

<item 34>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
), got every player invested in our characters and each other (even in a group of total strangers), and allowed the whole table to lean into both the strange
and ridiculous nature of life. This year, it was just the kind of joy that I needed.

2023 was a personal tabletop gold mine. We finally got Brennan Lee Mulligan’s first official 5e homebrew; I discovered tabletop meetup hub and game master meat
market StartPlaying.Games, where you can, as I have, hire a handsome Scottish man to D&D dominate you and your friends every week; and I played a ton of my new
favorite party game, As You ’Wich. It’s a dead-simple hand-builder that, for me at least, boils down to convincing your nephews to eat disgusting sandwiches
(hair, mustard, and broken glass on marble rye, absolutely not toasted).
</item 34>

<item 35>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Is Warlock the best class for you in BG3?

Warlocks sacrifice the number of spells that they know for sheer power. Every Warlock spell slot is the same — the highest level available — and they recharge
on a short rest instead of a long rest. Warlocks aren’t as versatile as other caster classes, but what they do, they do well.

Warlocks are a great class for you if you want to play a dedicated caster class, but don’t want to deal with all the prep work that classes like Wizards
require. You’ll still get to use (a lot of) magic, but you won't have to manage a binder full of spells every morning.
</item 35>

<item 36>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Best Sorcerer leveling build

This build is based off Lexyu’s Sorcerer leveling guide at IcyVeins, and is purely for leveling: i.e., allocating your 58 skill points between levels 1 and 50.
If you want more detailed information about the item affixes you want to look for on your gear, or which other Legendary powers are great for your build, check
out the original IcyVeins guide.

Best skills for leveling

This Sorcerer build guide is going to see you harness the awesome power of Lightning to speed through the leveling process. There are a few options for Core
skills here, but we’re specifically going to be showing you Lexyu’s Chain Lightning variation. In this build, you’ll also use Frost Bolt, Teleport, Frost Nova,
Lightning Spear, and Ice Armor.

Here are the best skills for a Sorcerer leveling build, organized by the exact order you should purchase them in — bearing in mind that some skills require you
to allocate a number of skill points in the tree in order to unlock them. As such,
</item 36>

<item 37>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
wasn’t consigned to the newcomers, though. Larian Studios, fresh off two years of early-access development and riding the reputation it had garnered from
Divinity: Original Sin 2, saw fit to release a role-playing game in which you can kill off nearly every main character the moment you meet them. Remedy
Entertainment — let’s be honest, this group has always been strange — made a sequel that’s equal parts horrifying, hilarious, fun, and fabulous. And Nintendo?
Well, Nintendo had another banner year. That’s no surprise. The real surprise? It finally let go, and let players toy with the digital molecules of its most
revered series. More on this below.

As the year comes to a close, it’s intoxicating to see developers of all sizes, in every genre, with every tier of budget, mining the depths of interactive
design, branching this way and that as they follow their respective veins of gold. They’re nowhere near the bottom of that particular expanse, of course — and
that’s a heartening thought.
</item 37>

<item 38>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Our campaign is called The Heartsguard Saga, and it’s based off Deven Rue’s map of a place called Euphoros. My players are at around level 15 and I frequently
give them a variety of amazing and unique gear that I’ve dreamed up all on my own... which they go on to forget they have, and never use. We used to play in
person, but since the pandemic I run the game virtually online from my house. I put a camera hovering over my gaming mat, and make a labeled grid for the
players so they call out, “Move me from A5 to B10.” During important rolls, I even make my players face their camera toward their dice for added suspense!

Petrana Radulovic, entertainment reporter

I’d wanted to play Dungeons & Dragons since I was in high school, but I never knew how to get into it.
</item 38>

<item 39>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
(The Great Old One exclusive) Expeditious Retreat Gain Dash immediately and as a bonus action on each of your turns until this spell ends. Faerie Fire All
targets within the light turn visible, and Attack Rolls against them have advantage. (The Archfey exclusive) Hellish Rebuke React to your next attacker with
flames that deal 2-20 Fire damage. Hex Make your attacks deal an addition 1-6 Necrotic damage to the target and give it Disadvantage on an Ability of your
choosing. Protection from Evil and Good Protect an ally against the attacks and powers of aberrations, celestials, elementals, fey, fiends, and undead. Tasha's
Hideous Laughter Leave a creature Prone with laughter, without the ability to get up. (The Great Old One exclusive) Sleep Put creatures into a magical slumber.
Select targets up to a combined 24 hit points. (The Archfey exclusive) Witch Bolt Link yourself to a target with a bolt of lightning.
</item 39>

<item 40>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Each Domain has different spells, actions, and even proficiencies associated with hit. Broadly, Life Domain focuses on healing, Light emphasizes Radiant and
Fire magic (in and out of combat), Trickery is good for sneaking, Knowledge gets extra Skills out of combat and brain-based spells, Nature is flavored a little
more like a Druid, Tempest is a little more combat and Thunder damage-focused, and War is all about combat.

Light Domain is the best subclass for a Cleric. Consult our Cleric build guide for more details on why, and to see which skills and feats to pick.

Best support class: Bard

Bards are a specialized kind of caster class that focus more on performance and charm than a weighty list of spells. Bards only get Light Armor and focus on
Charisma as their primary stat, so they rarely belong on the front line of a fight.
</item 40>

<item 41>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
this build, you’ll want to take One-Handed Mace in your Technique.

Vampiric Powers for Double Swing Barbarian build

Season 2 adds the new Vampiric Powers, which allow you to slot five additional abilities into your character to give you new effects. There are 22 total
Vampiric Powers, and each are class agnostic, meaning they’re not unique to any one class.

AaronActionRPG recommends these five Vampiric Powers:

Anticipation

Prey on the Weak

Ravenous

Domination

Metamorphosis

To insert those powers you’ll need to have these Pacts available on your armor:

8 Ferocity

3 Divinity

2 Eternity

Legendary Powers and Uniques for Double Swing Barbarian build

To make this build work, you’re going to need powerful Legendary Powers and Unique items to augment your build.

AaronActionRPG recommends these items:

Helm : Weapon Master’s Aspect

: Weapon Master’s Aspect Chest : Aspect of Numbing Wrath (dungeon drop)

: Aspect of
</item 41>

<item 42>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Finding the best class really just boils down to how you want to play. Even when playing solo, you’ll still manage the rest of your party, so you’ll get to
experiment with multiple classes at once.

BG3 classes have strengths outside of combat

Your character’s class is mostly about combat, but the underlying Ability Score that it relies on affects the other skills you’ll use to interact with the
world. Baldur’s Gate 3 assigns your Ability Scores automatically (though you can tweak it in the character creator), so each class’s primary ability will be
your highest score.

Bards and Sorcerers are both Charisma-based classes, for example. Their high Charisma will also make them good at skills like Persuasion, Deception, and
Intimidation. Out of combat, those classes will be better at talking their way out of (and into) trouble and lying. Wisdom-based classes like Clerics or Druids
will be more insightful and perceptive of their surroundings.
</item 42>

<item 43>
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
</item 43>

<item 44>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
There’s a lot of overlap in the classes — especially as you level up. And that means there’s no easy way to say one class is best (or even better). Instead,
focus on what you want to do (or how you want to do it), and go from there.

We’ll list them all below ranked (subjectively) by how difficult they are to play. We’ve put the caster classes toward the more difficult end simply because
they’ve got more to manage with their spells and spell slots — punching things is easier than memorizing spells.

Best melee class for beginners: Fighter

Nothing against Fighters here, but they’re the easiest class to wrap your head around. You’re not going to need to manage spells or Ki Points. Instead, you’ll
see a baddie and then you’ll stab said baddie.

Fighters start out as a very straightforward melee class. With high strength, they get all armors and shields, and choose a method of combat to focus on (and
get buffs to).
</item 44>

<item 45>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Triangle Agency

“I laughed so hard my face hurt.” —me, 10 minutes after my first session of Triangle Agency

I love a game with style and a clear, fun point of view, so I’ve been a fan of Triangle Agency since I picked up its player guide in beta form earlier this year
(to prep for writing a published adventure for the game). Written in the form of a cheerful, slightly threatening employee handbook, the Field Agent Manual
guides you through the creation of your supernaturally powered character, then gets you ready for your role finding and containing Anomalies that threaten the
nature of reality. After your team’s morning meeting, of course.

Beyond the style of the materials, though, playing the game was an absolute blast. The mix of mundane life, weird occurrences, and corporate BS created a
fast-paced game experience that was playful and easy to understand (just count those threes!
</item 45>

<item 46>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—JL

20. Chants of Sennaar

Developer: Rundisc

Where to play: Nintendo Switch, PlayStation 4, Windows PC, and Xbox One

In Chants of Sennaar you climb up a kind of Tower of Babel, and your job is to decipher the distinct language each group of people speaks, eventually aiming to
translate fluidly from one language to another. You do so by matching pictographic symbols — the characters for each language — to images of the nouns, actions,
and concepts in a large dictionary-style book.

It sounds complicated, but it’s wonderfully fun because the mechanics are so simple. You explore the isometric world — which is rendered in a gorgeous
cel-shaded style — witnessing interactions between cultures and attempting to puzzle out the meaning of their written words. The game parcels out words for you
to assign meaning to in little packets, to avoid overwhelming you. By the end, you’ll have translated numerous languages, and scaled your way to the top of the
tower. Perhaps you’ll even change the tower itself.
</item 46>

<item 47>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Best class for social interactions: Sorcerer

Sorcerers get their magic from within themselves. Sorcerers are the most flexible and varied in the spells they can use. Their primary Ability is Charisma, so
they’re also good at dialogue-heavy encounters out of combat. Sorcerers don’t get any armor, so you’ll have to keep them out of melee combat most of the time.

You’ll pick your Sorcerer subclass when you create your character from the Wild Magic, Draconic Bloodline, and Storm Sorcery subclasses. Wild Magic leans into
the chaotic and untamed nature of magic, Draconic Bloodline gives a dragon-based flavor to your magic, and Storm Sorcery lets you Fly whenever you cast a spell.

Draconic Bloodline is the best subclass for a Sorcerer. Consult our Sorcerer build guide for more details on why, and to see which skills and feats to pick.
</item 47>

<item 48>
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
</item 48>

<item 49>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Best healer class: Cleric

Clerics are the other side of the Paladin’s mix of magic and melee. They’re not quite as frontline in combat with only Light and Medium armors, but they get
more (and more powerful) spells. Clerics also get an extra boost with Channel Divinity that gets them an extra (Divine) magic ability like Turning Undead to
drive away zombies or Guided Strike that gives a +10 to Attack Rolls.

When you make a Cleric, you choose a subclass (Domain) from a list of seven: Life, Light, Trickery, Knowledge, Nature, Tempest, and War. Each Domain has
different spells, actions, and even proficiencies associated with hit.
</item 49>

<item 50>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
There are four main ideas that all Barbarian builds stem from:

Fury dumping (most popular in Diablo 4’s pre-season)

Weapon swapping

Bleed

Berserk

Fury dump builds make use of the Unbridled Rage Key Passive, which causes your Fury spenders to cost double the Fury but deal more than double the damage. These
kinds of builds obviously require very steady Fury generation, as it’s easy to starve yourself. Berserk builds that use the Unconstrained Key Passive are all
about keeping your Barbarian in the Berserk stage at all times, which nets you a hefty bonus. All Barbarian builds will likely use some skills that make them go
Berserk, the Unconstrained Key Passive just enhances the effect.

Bleed builds use the Gushing Wounds Key Passive to create a critical strike heavy build that’s great for whittling down groups of enemies with damage over time
and explosions. Weapon swapping builds offer what is almost certainly the most unique playstyle for Barbarians and takes advantage of the Walking Arsenal Key
Passive.
</item 50>

<item 51>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
(ranks 1-3) Counteroffensive (ranks 1-3) Defensive Stance (ranks 1-3) Slaying Strike (ranks 1-3) Aggressive Resistance Prolific Fury

Best Weapon Expertise setup for leveling

Once you unlock the Weapon Expertise system for your Barbarian, you’ll want to place the two-handed sword in your Technique slot. This will cause all of your
Hammer of the Ancients damage to make enemies Bleed.

Best Legendary Aspects for Barbarian leveling build

When you first start your new character, complete these three dungeons first in order to collect their Aspects. Then place them on a piece of gear (ideally
jewelry, when possible). Just keep in mind that you should replace these Legendaries with new versions using the imprint system every few levels.

Aspect of Ancestral Force (Sunken Ruins dungeon in Scosglen) — Hammer of the Ancients quakes outwards,
</item 51>

<item 52>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Here is how you should spend your Skill Points for this build:

Fire Bolt / Enhanced Fire Bolt / Glinting Fire Bolt

Fireball / Enhanced Fireball / Greater Fireball

Potent Warding (rank 1-3)

Devastation

Elemental Dominance (rank 1-3)

Flame Shield / Enhanced Flame Shield / Shimmering Flame Shield

Teleport (rank 1-4) / Enhanced Teleport / Shimmering Teleport

Ice Armor/Enhanced Ice Armor

Glass Cannon (rank 1-3)

Precision Magic (rank 1-3)

Align the Elements

Mana Shield (rank 1-3)

Protection (rank 1-3)

Firewall (rank 1-5) / Enhanced Firewall / Wizard’s Firewall

Inner Flames (rank 1-3)

Crippling Flames (rank 1-3)

Inferno/Prime Inferno/Supreme Inferno

Fiery Surge

Endless Pyre

Combustion

Enhancement slots for Sorcerer Firewall build

For this build,
</item 52>

<item 53>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
The Warlock is one of twelve classes in Baldur’s Gate 3. Warlocks are extreme knowledge-seekers who are willing to form a pact with otherwordly beings in order
to gain ancient secrets that’ll bolster their magical arsenal.

To deal with such powerful creatures, Warlocks need to have strong mental power. They need to be able to make a deal with the devil and survive. As such, their
primary ability score is Charisma.

There’s no single best class in Baldur’s Gate 3, and the game gives you as many opportunities for creative problem solving as it does for punching. Finding the
best class really just boils down to how you want to play. Even when playing solo, you’ll still manage the rest of your party, so you’ll get to experiment with
multiple classes at once.

Read on to find out everything you need to know about the Warlock class in Baldur’s Gate 3, and learn all about the best Warlock subclass, feats, and build.
</item 53>

<item 54>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
Best Barbarian leveling build

This build is based off Lexyu’s Barbarian leveling guide at IcyVeins, and is purely for leveling: i.e., allocating your 58 skill points between levels 1 and 50.
If you want more detailed information about the item affixes you want to look for on your gear, or which other Legendary powers are great for your build, check
out the original IcyVeins guide.

Best skills for leveling

Most Barbarian leveling guides you’ll find make use of Hammer of the Ancients as your primary skill, as it deals massive damage while remaining very cheap to
cast. In this build, you’ll also use Frenzy, Death Blow, Rallying Cry, Leap, and Wrath of the Berserker.

Here are the best skills for a Barbarian leveling build, organized by the exact order you should purchase them in — bearing in mind that some skills require you
to allocate a number of skill points in the tree in order to unlock them. As such, you’ll need to unlock new skills before enhancing existing
</item 54>

<item 55>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
This guide assumes that you’re over level 50, have access to all 58 skill points, and are making progress on your Paragon Boards. If you want more detailed
information about the item affixes you want to look for on your gear, or why these Legendary powers are great for your build, check out the original MaxRoll
guide.

Best skill points for Sorcerer Firewall endgame build

This build utilizes Firewall as its core skill and main means of dealing damage. It also uses Fire Bolt, Inferno, Teleport, Ice Armor, and Flame Shield.

Since you need all 58 points for this build, the order you select these skills in doesn’t matter as long as you have the pre-requisite points in the earlier
Skill Tree tiers.
</item 55>

<item 56>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Furthermore, in the process of putting a game together, I found a glut of free online tools from dungeon generators to NPC character sheet generators that made
game prep a breeze. I find myself coming back to this game again and again with each game session only giving me ideas for future campaigns. I know I’m not
alone in this feeling given the number of Mörk Borg-compatible games that have sprung up this year, from Pirate Borg to Chris Lockey’s upcoming Abyss of
Hallucinations.

I’m Sorry Did You Say Street Magic

Madison Durham, writer and journalist, staff writer at Reviewed

To me, some of the most joyful moments in tabletop gaming come from the experience of creative collaboration — the magic of making something new with friends,
be it a narrative, a character, a world, or in this case, a city. I’m Sorry Did You Say Street Magic by Caro Asercion is a GM-less city-building game, played
between a group of two to six people with a stack of index cards and your favorite writing utensils.
</item 56>

<item 57>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
(The Archfey exclusive) Witch Bolt Link yourself to a target with a bolt of lightning. Deal an additional 1-12 Lightning damage each turn by activating it.

Actions

Warlocks have one action, “Fey Presence,” which is exclusive to The Archfey subclass.

Features

Warlock Spell Slot — Used to cast Warlock spells, which can be regained by taking a short rest

Proficiencies

Weapons — Simple Weapons

— Simple Weapons Armor — Light Armor

— Light Armor Saving Throws — Wisdom and Charisma

— Wisdom and Charisma Skills — Choose two skills from Arcana, Deception, History, Intimidation, Investigation, Nature, and Religion

Warlock subclasses

The Warlock has a total of three subclasses, which are unlocked at at the beginning of the game. During character creation, you must pick the otherworldy patron
that you’ve decided to serve.

The Fiend

Warlocks with The Fiend subclass have struck a deal with a hellish fiend who strives for corruption and destruction.
</item 57>

<item 58>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Should you play a custom or an Origin character in BG3?

Baldur’s Gate 3 gives you the option to play as an Origin character, meaning you can just choose one of the NPCs from your party and play as them right out of
the gate. It’s a perfectly fine way to play if you really don’t want to go through the process of picking out a race and background for your character.

That said, you’ll also be limited to that character’s predetermined story. It’s a bit more work at the beginning, but you’ll have a lot more freedom to
experience the entire story of Baldur’s Gate 3 if you make a custom character. With all that in mind...

Which BG3 class (and subclass) is best for you?

Distinctions like caster vs. puncher or melee vs. ranged don’t really work for Baldur’s Gate 3 (or D&D). There’s a lot of overlap in the classes — especially as
you level up.
</item 58>

<item 59>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—Ana Diaz

49. Lies of P

Developer: Neowiz Games

Where to play: Mac, PlayStation 4, PlayStation 5, Windows PC, Xbox One, and Xbox Series X

Yes, Lies of P is a Dark Souls mixed with Pinocchio, and that’s a questionable elevator pitch from the outset.

In the years leading up to Lies of P, “Pinocchiosouls” was more of a running joke than anything — this profane idea that you can take any world and slap some
Dark Souls into it to get people mildly interested. But once you’re in the game, eliminating bosses left and right with your sweet parry moves, you’ll quickly
find yourself entirely unbothered by how strange Lies of P initially seemed. And you’ll start telling your partner things like “I have to go back to Geppetto to
upgrade my puppet body” like it’s a perfectly normal task to assign yourself on a Tuesday afternoon.
</item 59>

<item 60>
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
</item 60>

<item 61>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Spell Sniper. You learn a cantrip, and the number you need to roll a Critical Hit while attacking is reduced by 1. This effect can stack.

For more information on the classes in Baldur’s Gate 3, consult our overviews on the Barbarian, Bard, Cleric, Druid, Fighter, Monk, Paladin, Ranger, Rogue,
Sorcerer, and Wizard classes. If you’re stumped by the choice, consult our guide on what class to start with.
</item 61>

<item 62>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—PV

7. Street Fighter 6

Developer: Capcom

Where to play: PlayStation 4, PlayStation 5, Windows PC, and Xbox Series X

After getting knocked down — by a self-inflicted punch, no less — with the feeble Street Fighter 5, Capcom has hit the gym and returned stronger than before
with Street Fighter 6. We rightfully called its latest Street Fighter the “ultimate fighting game toolbox” in Polygon’s review; from a robust single-player mode
to solid online modes to a cast of memorable new and returning World Warriors, Street Fighter 6 is Capcom at its most confident.

For newcomers and the lapsed Street Fighter fan wary of jumping into online play, Capcom delivered World Tour mode, a robust, single-player, RPG-lite
beat-’em-up in the vein of Sega’s Yakuza games. In World Tour, players hit the streets of Metro City where, hilariously, everyone in town not only knows how to
fight, but relishes impromptu fisticuffs with strangers. It’s a matter of local pride.
</item 62>

<item 63>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Picking the best class in Baldurs Gate 3 is one of the most important decisions you’ll make in the game — and you have to do it right at the start, with minimal
context to work off.

Your class mostly determines how your character will fight in Baldur’s Gate, but with 12 classes — and 46 subclasses total across them all — it’s not really an
easy decision. Here’s what to know about finding the best class (and the best subclass) for you in Baldur’s Gate 3.

All BG3 classes are ‘good’

The classes in Baldur’s Gate 3 — and in pen-and-paper D&D, for that matter — are all fairly well-balanced. There’s no single best class (barrelmancer aside) in
Baldur’s Gate 3, and the game gives you as many opportunities for creative problem solving as it does for punching. Finding the best class really just boils
down to how you want to play.
</item 63>

<item 64>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Best magic class for beginners: Warlock

Warlocks are casters that made a deal with a higher power for their magic. They play a lot like Sorcerers, but with a slight emphasis on more academic Skills
and fewer spells. Warlocks get Light Armor, so they’re a little less squishy than Sorcerers.

Like Sorcerers, you’ll pick your Warlock subclass during character creation. You’ll choose from The Fiend, The Great Old One, and The Archfey subclasses that
are all themed around where your magic comes from — and determine some of the spells you can use. The Fiend Warlocks get their magic from a demon-like fiend and
have hell-themed spells, the Great Old One is more Lovecraftian horror-themed, and the Archfey draws power from a powerful Fey being.

Great Old One is the best subclass for a Sorcerer. Consult our Sorcerer build guide for more details on why, and to see which skills and feats to pick.
</item 64>

<item 65>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Frost Nova (rank 2-3)

Best Enchantment setup for leveling

The Enchantment slot is key to a ton of your damage as a Sorcerer. You’ll gain two slots as you level, here’s which spell you want to put in each slot:

Level 15: Fireball

Level 30: Chain Lightning

Best Legendary Aspects for Sorcerer leveling build

When you first start your new character, complete these three dungeons first in order to collect their Aspects. Then place them on a piece of gear (ideally
jewelry, when possible). Just keep in mind that you should replace these Legendaries with new versions using the imprint system every few levels.
</item 65>

<item 66>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Wizards don’t get any armor and are limited to just daggers and quarterstaffs (quarterstaves?) in combat, so they’re going to rely on magic to do any damage.

Your Wizard subclass is chosen when you create the character. You’ll choose from eight schools of magic: Abjuration, Conjuration, Divination, Enchantment,
Evocation, Illusion, Necromancy, and Transmutation. Each subclass gets an extra ability themed to the school of magic. Broadly speaking, Abjuration is about
protection and defense, Conjuration is about creating something from nothing, Divination gets to foresee (and change) rolls, Enchantment is about charming and
controlling minds, Evocation shapes and controls spells, Illusion is about trickery, Necromancy controls life and death, and Transmutation focuses on alchemy
and changing the world around you.

Evocation is the best subclass for a Wizard. Consult our Wizard build guide for more details on why, and to see which skills and feats to pick.
</item 66>

<item 67>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
And there is already loads of content to support that kind of time investment. —RG

Related Diablo 4 activates and frustrates my lizard brain

27. Hitman World of Assassination: Freelancer mode

Developer: IO Interactive

Where to play: PlayStation 4, PlayStation 5, Windows PC, Xbox One, and Xbox Series X

Hitman World of Assassination’s Freelancer mode, which debuted in January and puts a roguelike twist on Agent 47’s globe-trotting murder-for-hire missions,
isn’t for the faint of heart. It demands constant improvisation, to a degree that can challenge even veteran Hitman players. Repeated failures can make it feel
more frustrating than fun.

But as every fan of roguelikes knows, these high-stakes experiences have the capacity to deliver a sense of exhilaration like nothing else. Pulling off multiple
daring kills, hiding the evidence (or going out with a bang), and trying to make it out alive with all the gear you brought into the mission — it’s tense and
thrilling, every time.
</item 67>

<item 68>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
The idea here is that you’re constantly swapping between your skills and weapons to give yourself massive damage buffs.

When making a build, you want to have one of these Key Passives in mind and choose your skills accordingly. Thankfully skills that share certain aspects (like
Bleed) are color-coded on the skill tree, and you can even use the in-game search function to find them all. Other Barbarian builds let you take advantage of
mechanics like Thorns or Shouts. But they still typically end with one of these Key Passives.

Looking for other Diablo 4 builds? We have explainers on the Sorcerer, Rogue, Druid, and Necromancer classes.
</item 68>

<item 69>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Most of their magic is nature- or animal-themed and they get Skills to match. What they lack in armor proficiencies — they only get Light and Medium Armor —
they make up for with the ability to shapeshift into a badger, wolf, spider, or cat.

At level 3, Druids pick from the Circle of the Land, Circle of the Moon, and Circle of Spores subclasses. Circle of Land gets you some extra magic and the
ability to cast more spells throughout the day, Circle of the Moon adds a bear to the repertoire of animal shapes you can shift into, and Circle of Spores adds
in some extra fungus- and decay-based magic.

Circle of the Moon is the best subclass for a Druid. Consult our Druid build guide for more details on why, and to see which skills and feats to pick.

Best class for stealth: Rogue

Rogues are Dexterity-based, sneaky fighters. They only get Light Armor, so they have to focus on staying unseen and quick.
</item 69>

<item 70>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
For weapons, Barbarians get a wide range of heavy hitters like battleaxes and warhammers. Every Barbarian gets the Rage ability that lets them hit even harder
and reduce damage during combat (and only during combat).

When you hit level 3, you get to choose from the Wildheart, Berserker, or Wild Magic subclasses. Wildheart lets you modify your Rage ability with animal-themed
tweaks like healing, jumping, or a stampede. Berserker is a lot more straightforward and focuses on hitting (extra) things and throwing. And Wild Magic adds a
magical flair to your Rage with a random magical effect and improved saving throws against magic for you and your allies.

Berserker is the best subclass for a Barbarian. Consult our Barbarian build guide for more details on why, and to see which skills and feats to pick.

Best class for magic: Druid

Druids are the nature-based casters of Baldur’s Gate 3.
</item 70>

<item 71>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Your second most important stat kind of depends on how you play outside of combat, but Constitution is also a good option for the hit points — Warlocks, like
Sorcerers, have a tendency to be glass cannons. Make Charisma your highest stat and Constitution your second highest.

You’ll pick your Warlock subclass at level 1, but the pact you choose here is actually less important than the Eldritch Invocation you get at level 2 — that
said, go with the Great Old One for the most broadly useful features. Your Eldritch Invocation powers up your Eldritch Blast spell — the bread-and-butter spell
for Warlocks — and turns it into an Agonizing Blast.

Best Warlock feats in BG3

Every four levels, you’ll get the option to either increase your stats or choose a Feat. Feats are special talents that add features to your character. If
you’re happy with your stats and start taking Feats, Warlocks can benefit from:

Actor .
</item 71>

<item 72>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Their best Ability is Dexterity, so their combat is more martial arts than swordplay. Monks don't get armor and only have a limited set of weapons available,
but they make up for it with Unarmed Strikes that deal extra damage. Monks take a little more finesse to play than Fighters.

Monk subclasses unlock at level 3 and include Way of the Four Elements, Way of the Open Hand, and Way of Shadow. Way of the Four Elements mixes in spells with
your Monk abilities (and uses renamed spells from the caster classes). Way of the Open Hand expands your attacks to include things like knocking down enemies or
pushing them with extra melee attacks. And Way of Shadow makes your Monk sneaky and stealthy (think of it as a ninja subclass).

Way of the Open Palm is the best subclass for a Monk. Consult our Monk build guide for more details on why, and to see which skills and feats to pick.
</item 72>

<item 73>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Conceited, or Storm Swell (dungeon drop)

: Edgemaster’s (dungeon drop), Conceited, or Storm Swell (dungeon drop) Focus : Edgemaster’s (dungeon drop), Conceited, or Storm Swell (dungeon drop)

: Edgemaster’s (dungeon drop), Conceited, or Storm Swell (dungeon drop) Amulet : Control (dungeon drop)

: Control (dungeon drop) Ring 1 : Prodigy’s (dungeon drop) or Retribution (dungeon drop)

: Prodigy’s (dungeon drop) or Retribution (dungeon drop) Ring 2: Engulfing Flames or X’Fal’s Corroded Signet (Unique)

If you want to get into the nitty gritty of stat priorities, refer to Northwar’s original guide.
</item 73>

<item 74>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
The best deals of the year were once known to occur during Black Friday and Cyber Monday, which in 2023 take place on Friday, Nov. 24, and Monday, Nov. 27,
respectively. Not anymore!

In 2023, the best Black Friday deals kicked off on Nov. 17 at Amazon, Best Buy, Walmart, and other retailers. Instead of hosting all of the doorbuster deals on
one day right after Thanksgiving, retailers see the value in maximizing the amount of time you have to shop for products before Black Friday. And I’ve gotta
say, it’s quite nice to not feel rushed while shopping.

In addition to presenting you with the best board game deals and price drops on gaming monitors, plus deals on SSDs and microSD cards, there are some bits of
info and wisdom I’ve organized into sections below that might help you feel more informed as you begin to shop for the holidays.
</item 74>

<item 75>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
When making a Lightning build, you’ll end up using the Overflowing Energy Key Passive or the Vyr’s Master Key Passive, depending on if you’re in melee range or
not.

Ice builds specialize in slowing enemies down, making it easier to survive their attacks. Ice builds will take advantage of either the Avalanche Key Passive or
the Shatter Key Passive, depending on how much your build relies on crowd control

Fire builds are all about big explosions and burning your enemies. For these builds, you’ll typically use either the Combustion Key Passive for burning enemies
or the Esu’s Ferocity Key Passive for chunky critical strikes.

When making your own build for a Sorcerer, you’ll likely end up sticking to one element for your main damage sources. But that’s not the most important thing
you need to remember. Sorcerers die very, very easily in Diablo 4, and they have the weakest defensive stats by far — however, they can deal a massive amount of
damage when built correctly.
</item 75>

<item 76>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
you’ll want to use these spells for your two Enchantment Slots:

Firewall Enchantment

Fireball

Vampiric Powers for Sorcerer Firewall build

Season 2 adds the new Vampiric Powers, which allow you to slot five additional abilities into your character to give you new effects. There are 22 total
Vampiric Powers, and each are class agnostic, meaning they’re not unique to any one class.

Northwar recommends these five Vampiric Powers:

Anticipation

Prey on the Weak

Ravenous

Flowing Veins

Metamorphosis

To slot those powers you’ll need to have these Pacts available on your armor:

9 Ferocity

5 Divinity

4 Eternity

Legendary Powers and Uniques for Sorcerer Firewall build

To make this build work, you’re going to need powerful Legendary Powers and Unique items to augment your build.
</item 76>

<item 77>
Diablo 4 season 2’s biggest draw is its quality-of-life updates (Polygon, 2023-10-04T20:01:00+00:00)
As for gems, these items will no longer take up bag space, as they’re now considered crafting materials. This change comes alongside a massive upgrade to the
Stash itself. Stashes now appear in more locations — including all minor towns with a Waypoint in them — and players can search their Stashes for specific items
and affixes.

These are just a handful of the major changes coming to Diablo 4 with Season of Blood. Shely and Piepiora told us that the focus on quality-of-life upgrades was
largely due to season 2 being the first season where they and the rest of the Diablo 4 team could really listen to feedback. Because of how close Season of the
Malignant launched after the release of Diablo 4 (just over a month), that season’s content was largely finished before players had really gotten to dive into
the action-RPG. However, with season 2, Blizzard has been able to react more to player feedback and finally create some sweeping, specific changes that players
have been requesting.
</item 77>

<item 78>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
After more than a decade of going on and on about the “board gaming renaissance,” it’s safe to say that tabletop role-play is finally having its moment in the
spotlight. But while Dungeons & Dragons’ OGL fiasco has clearly lit a fire under many players eager to make a change, the fact of the matter is that a sizeable
cohort of writers and designers have been toiling away at excellent games for years now. Their tireless efforts have yielded a bumper crop of excellent, some
might say genre-defining, TTRPGs. Just like board games, a few modern classics have also cropped up: Look no further than The Quiet Year and The One Ring, both
making their second appearance on this list.

Just as before, Polygon asked nearly two dozen writers, designers, presenters, actors, and personalities from around the world of tabletop gaming to share with
our readers the TTRPGs that made an impact on them and their players this year. Here’s what we found.
</item 78>

<item 79>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Players take on the role of soldiers and other hangers-on just trying to make it out alive. The heady mix of dice-driven gunplay and a playing-card-style
encounter creation makes it easy to run. But the fun begins even before that with character generation, which uses old school randomized tables to simulate an
entire military career. If you’re looking to get your video gaming buddies around the table for something that doesn’t require a headset or shouting down
pre-teens in a glitchy public lobby, the all-inclusive starter set comes highly recommended.

Project ECCO

It’s a near-impossible task to get a consistent group together, so solo TTRPGs have become my primary method of gaming. However, the barrier for me is finding a
game with a balance of narrative and mechanics that isn’t overwhelmingly crunchy, but also isn’t open-ended to the point of choice paralysis. For me, Project
ECCO found that balance.
</item 79>

<item 80>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
I played Marvel’s Avengers, too, and this isn’t that. It might seem like it’s going to be at first, because Midnight Suns makes the grave error of introducing
Iron Man and Doctor Strange as its tutorial characters, and these two might just be the most irritating characters in the entire video game. (I have beaten the
game, so I am allowed to make this call.) You must press on and give Midnight Suns time to win you over. Because it has so, so much more to offer than it may
appear in its first few hours.

Picture the romance and humor of Fire Emblem: Three Houses, combined with the high-stakes tactical battles of XCOM 2 — that’s what Midnight Suns becomes in its
mid-game and endgame. It’s a card-based strategy game, and each hero has their own customizable deck. I started off favoring Captain Marvel, Magik, and Blade,
simply because their moves and hilarious dialogue kept me entertained, but I soon realized that every single character has something exciting or unexpected to
bring to the battlefront.
</item 80>

<item 81>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Almost every champion has felt viable throughout the year — no small feat for a game with over 140 playable characters — and it’s led to great fun and variety
on the solo queue ladder and in professional play, where an exciting Worlds just wrapped up.

But there’s another reason League had an outstanding 2023: Arena, a new game mode introduced during the game’s summer event. Arena is a 2v2v2v2 battle mode with
fast-paced chaos, using League’s roster of champions in a more approachable and containable setting (and with less rage-inducing teammates). The mode was
removed after the conclusion of the summer event, but is reportedly returning soon. It can’t possibly come soon enough; I know how I’ll be spending a good chunk
of my winter. —Pete Volk

Related Arcane is officially League of Legends canon now

41.
</item 81>

<item 82>
Diablo 4 season 2’s biggest draw is its quality-of-life updates (Polygon, 2023-10-04T20:01:00+00:00)
Teleporting to Nightmare Dungeons is one thing, but players will still need to ride their horses around the world if they want to reach events like Helltides
and the new seasonal activity, Blood Harvest. Thankfully, mounts have also been improved in season 2. Horses are now faster, and their dashes will allow them to
break through barricades in the roads. Shely told us that the mounts should be more intelligent as well, and they won’t get stuck on objects in the world as
easily as before.

Two of the other biggest pain points — gems and Renown — are also getting addressed in this season.

The grind to accrue Renown — the resource players get for completing activities in each area — is no longer something players will need to partake in every
season. Much like the Altars of Lilith change in season 1, players who’ve fully unlocked their Renown track with one character will never have to do it again.
</item 82>

<item 83>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Cannot recommend enough.

The Monster Overhaul

The Monster Overhaul, by Skerples, reimagines not just the monsters, but the very notion of what a bestiary can and should be. The book is divided into 20
categories, each containing 10 critters that hew to a theme. The categories are unusual: There is “Dragons,” of course, but also “Summer” and “A Wizard Did It.”
“Summer” monsters include the Froghemoth and Pyromancers. Some of these may sound similar to classic D&D monsters, others are entirely new. Tables galore help
build and flesh out encounters. “Summer” has a set of generic swamp hexes; other entries have lairs and dungeons. There is an entire flowchart table for
populating a megadungeon. Every page of this book is designed to make the reader think about monsters, how to make them feel new, or to recontextualize them, or
to simply subvert player expectations.
</item 83>

<item 84>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Thankfully, in the doldrums of 2020, someone I knew knew someone who knew someone (that’s literally what the connection was) who was interested in running a
game. That kicked off my D&D career, but was this year that everything finally clicked in my brain and it went from a hobby I enjoyed to one that I spend every
free minute of my spare time thinking about.

Part of it is finally having reliable groups. After some initial shuffling of members, in 2023 I’ve been blessed to have not one but two separate D&D groups,
both with enthusiastic members who are committed to at least trying to meet regularly. And yes, part of it is Baldur’s Gate 3, which helped me to understand
some of the more complicated D&D mechanics and just generally made me very excited about combat in a way I never was before.

I’ve still yet to actually have a game in person! My groups are spread out across North America, but I’m hoping to manifest an in-person session soon.
</item 84>

<item 85>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
In Alan Wake 2, everyone’s story is happening at once, branching out in endless directions, and we might not know the genre until it’s too late. It’s a
hypertext mystery, an ergodic game that molds itself to an audience raised online, though few computers are in sight. It’s a game for a world suspended in an
endless second act — forever wars, live services, franchises, infinite scrolls — endlessly searching for a conclusion. —JR

2. Baldur’s Gate 3

Developer: Larian Studios

Where to play: PlayStation 5 and Windows PC

Baldur’s Gate 3 hit at the perfect moment. Despite the handful of truly great computer role-playing games that have been released since the early 2000s (Larian
Studios itself made waves with its Divinity: Original Sin series in 2014 and 2017), the genre has largely remained a modern niche. But times have changed:
Dungeons & Dragons has undergone something of a renaissance, thanks to actual-play series like Critical Role and Dimension 20.
</item 85>

<item 86>
How queerness became calamity in Tears of the Kingdom (Polygon, 2023-10-19T19:00:00+00:00)
The change transforms queer materials into something more like cultural tchotchkes, further denying the queer subtext of trans masculinity and furthering the
sense of Orientalism pervading the Gerudo.

Tears of the Kingdom does introduce new armor sets that can play with expression, but none are explicitly feminine. Most are Zonai in origin, a fictional
ancient culture that borrows heavily from pan-Mesoamerican and Egyptian imagery. As such, it’s hard to read their skirts and body paint as feminine when these
were features of masculine presentation to their cultures. I guess there is the one-off Cece’s Hat, designed by a Hylian in Hateno, but it lacks a matching set
of top and bottoms. So, there’s nothing really replacing the gender nonconformity that the vai armor introduced in Breath of the Wild. But this would perhaps be
less notable if Link’s gender presentation wasn’t so ridiculed across Hyrule.

What pervades the land in place of calamity is not the evil of Ganon, but of gender deviance.
</item 86>

<item 87>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
If you forge a pact with a fiend, you’ll receive a feature exclusive to The Fiend called “Dark One’s Blessing,” which will grant you four temporary hit points
whenever you defeat a hostile creature.

The Fiend also has two exclusive spells — Burning Hands and Command.

The Great Old One

Those serving The Great Old One are bound to a mysterious being from the Far Realm, who might not even know you exist, The Great Old One will grant you a
feature exclusive to this subclass called “Mortal Reminder,” which will frighten a creature and any nearby enemies after you land a critical hit unless they
succeed a Wisdom saving throw.

The Great Old One also has two exclusive spells — Dissonant Whispers and Tasha’s Hideous Laughter.

The Archfey

If you choose The Archfey as your subclass, your patron is a lord or lady of the fey, a creature from before the time of mortal races.
</item 87>

<item 88>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Sorcerers are the magic-slinging wizards of Sanctuary in Diablo 4. But with each element they can harness bringing its own unique effect, it can be easy to get
overwhelmed when trying to design a Sorcerer build. That’s where we come in, with our suggestions of the best Sorcerer builds for Diablo 4 season 2.

Sorcerers are capable of harnessing lightning to deal with crowds, ice to keep the hordes at bay, and fire to burn the demons of hell. But they’re also able to
enchant their own spells with unique effects, intricately combining abilities for devastating results.

In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Sorcerer builds for season 2. If you need even more information on item affixes or other
more complex ideas, we’ve been sure to link out to the more detailed versions of these builds, as well as where to find their original creators.
</item 88>

<item 89>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
For more information on the classes in Baldur’s Gate 3, consult our overviews on the Barbarian, Bard, Cleric, Druid, Fighter, Monk, Paladin, Ranger, Rogue,
Sorcerer, Warlock, and Wizard classes.
</item 89>

<item 90>
Diablo 4 season 2’s biggest draw is its quality-of-life updates (Polygon, 2023-10-04T20:01:00+00:00)
The Echo of Duriel (who is level 100) will be the most difficult of the new bosses, but the reward will obviously be worth the struggle, considering he’s the
only way for players to farm Uber Uniques in Diablo 4. For players who have more in-game gold than time, Blizzard also confirmed in our interview that players
will be able to trade with one another to get the items needed to summon Duriel and the other bosses.

While these bosses are a form of milestone activity for players, Nightmare Dungeons will still exist as places to both farm for better items and push builds to
their limit. However, the dungeons are becoming much simpler and less grindy. The design team has reworked several Nightmare Dungeon affixes to make them less
obnoxious, and Nightmare Sigils now teleport players directly into the dungeon instead of to the door.
</item 90>

<item 91>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Northwar recommends these items:

Helm : Shared Misery (dungeon drop) or Godslayer Crown (Unique)

: Shared Misery (dungeon drop) or Godslayer Crown (Unique) Chest : Snowveiled (dungeon drop) or Raiment of the Infinite (Unique)

: Snowveiled (dungeon drop) or Raiment of the Infinite (Unique) Gloves : Edgemaster’s (dungeon drop), Conceited, or Storm Swell (dungeon drop)

: Edgemaster’s (dungeon drop), Conceited, or Storm Swell (dungeon drop) Pants : Disobedience (dungeon drop)

: Disobedience (dungeon drop) Boots : Ghostwalker (dungeon drop), Bounding Conduit (dungeon drop), Flickerstep (Unique)

: Ghostwalker (dungeon drop), Bounding Conduit (dungeon drop), Flickerstep (Unique) Wand : Edgemaster’s (dungeon drop), Conceited,
</item 91>

<item 92>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Minor Illusion Create an illusion that compels nearby creatures to investigate. Poison Spray Project a puff of noxious gas. True Strike Gain Advantage on your
next Attack Roll.

Spells

The Warlock starts with two spells from the Warlock spell list, two of which are exclusive to your subclass.

Baldur’s Gate 3 Warlock Spells Spells Description Spells Description Armor of Agathys Gain 5 temporary hitpoints and deal 5 Cold damage to any creature that
hits you with a melee attack. Arms of Hadar Prevent targets from using reactions. Burning Hands Each flammable target is hit with 3-18 Fire damage. (The Fiend
exclusive) Charm Person Charm a humanoid to prevent it from attacking you. You gain Advantage on Charisma Checks in dialogue. Command Command a creature to
flee, move closer, freeze, drop to the ground or drop its weapon. (The Fiend exclusive) Dissonant Whispers Frighten a creature: it will be easier to hit and
cannot move.
</item 92>

<item 93>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The second, just launched, is the wild Season of Discovery, which remixes and restructures the original WoW experience — interpolating staggered level caps,
shuffling class roles — in ways that might just change MMO design forever. WoW Classic is quietly, and paradoxically, where Blizzard is doing its most
forward-thinking work right now. —Oli Welsh

39. Fire Emblem Engage

Developer: Intelligent Systems

Where to play: Nintendo Switch

Fire Emblem Engage was designed for a very specific kind of sicko: one not particularly interested in the origin stories of a horde of teenagers, or the
politics of a bourgeoise academy, or what kind of tea a teacher prefers, but instead one obsessed with the endless minutiae of combat stats, weapon loadouts,
and team composition. I know this because I am one such sicko.

If you’ve read any of my reviews or essays on Polygon, then you know I prefer strategy games that can get out of their own way.
</item 93>

<item 94>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The gameplay didn’t just help me stick with the game, but instead allowed my excitement to bubble over every time I took on a new mission. —Ana Diaz

15. Cyberpunk 2077: Phantom Liberty

Developer: CD Projekt Red

Where to play: PlayStation 5, Windows PC, and Xbox Series X

Like the best DLC, Phantom Liberty does as much to expand on the base game as it does to reframe it. Alongside the sweeping Patch 2.0, which revamped Cyberpunk
2077’s role-playing progression systems, improved its enemy AI, and course-corrected a litany of other details, Phantom Liberty also adds a whole new district
to the dystopian world of Night City, complete with its own spy-thriller storyline.

The resulting Cyberpunk 2077 is not an entirely different beast than the one that was released in 2020, but it is a more evolved one.
</item 94>

<item 95>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
If you don’t want to end up dead more often than not, make sure to invest heavily in defensive skills while making your way toward one of the six Key Passives
we listed above.

Looking for other Diablo 4 builds? We have explainers on the Barbarian, Rogue, Druid, and Necromancer classes.
</item 95>

<item 96>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
It’s a matter of local pride.

The streets of Metro City are part training grounds, part beginning of a silly, epic worldwide adventure where you learn the basics of Street Fighter 6.
Untethered from any serious game narrative canon, Street Fighter 6 lets players of all skill levels and stripes have fun in World Tour mode.

Street Fighter 6’s approachability extends to its innovative new control scheme, an addition called Modern Controls. While the six-button layout from the very
first Street Fighter is still available, after multiple attempts at giving players a simplified control scheme, Capcom’s finally cracked it. Modern Controls are
not only comparatively easy to grasp — they’re pretty viable competitively.

But it’s the finely honed one-on-one fighting mechanics, governed by a streamlined set of meters and flashy new moves, that give Street Fighter 6 its longevity.
Thanks to a well-populated social space called the Battle Hub, and a rock-solid online infrastructure, there’s a robust community of other Street Fighter fans
to battle against on a daily basis.
</item 96>

<item 97>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
I can’t recommend it enough. Grab a copy of the PDF, turn down the lights, and start mapping your own haunted house!

The Luminant Age

I became an immediate fan of Luminant Age after being invited to be a contributor. What caught my eye at the start was the Four Humanities ancestries named
Blood, Clay, Silk, and Horn. This year, many of my Thursday mornings have been a whirlwind of weird as our playtest party encounter teleporting flickrats, heal
via esseweaving, and get hunted down by cultists in stryx bonemasques. The city we’re in, Ourichor, is built on giant oil-rig-like platforms in the open ocean.
There are no sun nor stars, but three moons... that you’re allowed to see.

Also, did I mention the predominant religion featured, the Angelites, memorialize their dead by preserving their eyeballs? You can even wear them as mourning
jewelry.
</item 97>

<item 98>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
What’s more, Larian had two years of player feedback with which to build its new masterpiece. That masterpiece launched in August, and it garnered the kind of
attention that catapults a game from niche interests to widespread acclaim.

Old CRPG fans finally came home, the Mass Effect generation has discovered a new kind of RPG to sink its teeth into, and D&D fans have found an exceptional
digital version of their beloved tabletop game to play alone or with their party in co-op.

Like a good DM, Baldur’s Gate 3 teases out story beats that feel personal to you, cleverly luring you into experiences that seem like they wouldn’t make sense
in anyone else’s campaign. Every act delivers quiet, memorable character moments and anxiety-inducing battles. All of these situations feel organic, allowing
Baldur’s Gate 3 to replicate, perhaps as closely as a video game can, Dungeons & Dragons’ best feature: the intoxicating sense that anything could happen at any
moment.
</item 98>

<item 99>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Warlock class features

Hit Points

Hit Dice — 1d8 per warlock level

1d8 per warlock level Hit Points at 1st Level — 8 + your Constitution modifier

— 8 + your Constitution modifier Hit Points at higher levels — 5 + your Constitution modifier per warlock level after 1st

Cantrips

The Warlock starts with two cantrips from the Warlock cantrip list.

Baldur’s Gate 3 Warlock Cantrips Cantrip Description Cantrip Description Blade Ward Take only half the damage from Bludgeoning, Piercing, and Slashing attacks
Bone Chill Prevent the target from healing until your next turn. An undead target receives Disadvantage on Attack Rolls. Eldritch Blast Conjure 1 beam(s) of
crackling energy. Friends Gain Advantage on Charisma Checks against a non-hostile creature. Mage Hand Create a spectral hand that can manipulate and interact
with objects. Minor Illusion Create an illusion that compels nearby creatures to investigate.
</item 99>

<item 100>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Like all great RPG supplements, The Monster Overhaul not only offers answers for these questions and more, it also teaches the reader how to continue answering
them long after these published tables and suggestions are exhausted. A monstrous achievement that should be on every GM’s shelf.

Bonus: Reach of the Roach God

Reach of the Roach God is the hardcover high-watermark of A Thousand Thousand Islands (ATTI), a series of zines, that presents a system-agnostic setting
inspired by the folklore of Southeast Asia. Odoyoq, the roach god, senses new ways it can insinuate itself into the lives of non-roachkind. The three schemes
are presented as both traditional adventures and as collections of facts — about the setting, about the people living there and their motivations, about the
desires of the god and the machinations of its agents. Events will quickly throw a small village, a monastery, and a necropolis into chaos. Two gazetteers and a
sourcebook on the roaches follow, giving the GM ample material to expand the campaign.
</item 100>
=== END ITEM id=16f4fb0452cc9986 ===

=== ITEM id=da5a9134df58bbe3 ===
Gold sentences and facts:
Sentence 0: Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
  Fact 0.0: Kenneth Walker III plays for the Seahawks.
  Fact 0.1: Kenneth Walker III took four carries against the Rams.
  Fact 0.2: Kenneth Walker III took four carries against the Rams before leaving.
  Fact 0.3: Kenneth Walker III left with an oblique injury.
Sentence 1: Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging injuries.
  Fact 1.0: Entering Week 12, Tee Higgins failed to suit up due to a nagging hamstring injury.
  Fact 1.1: Entering Week 12, Noah Brown failed to suit up due to a nagging knee injury.
  Fact 1.2: Entering Week 12, Treylon Burks failed to suit up due to a nagging concussion.
  Fact 1.3: Entering Week 12, Kadarius Toney failed to suit up due to a nagging ankle injury.
  Fact 1.4: Entering Week 12, Kadarius Toney failed to suit up due to a nagging hip injury.

Retrieved items:
<item 1>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 1>

<item 2>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
These are exactly the arguments that other search engine companies, like DuckDuckGo, Brave or Microsoft’s Bing, make when they accuse Google of being
anticompetitive by making deals with Apple and others. The DOJ doubled down on this, saying Google has become the monopoly it denounced years ago.

What does it all mean?

The case is expected to continue for several weeks, bringing to a head one of the biggest fights in tech antitrust since the U.S. took Microsoft to trial in the
1990s.

If the judge rules against Google, the outcome could look a lot like the Microsoft deal, in which the computer company was required to change its behavior and
share its APIs with third-party developers. Microsoft was also banned from making anticompetitive and exclusive deals with computer manufacturers.

Google might end up having to turn over all or most of the data it has collected to other search engines so they can improve their products and attract more
users. The DOJ has said that Google gets 16 times more data than Bing does everyday.
</item 2>

<item 3>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Bellshaw asked Cue to review some of Apple’s financial filings. Isn’t it true that the ISA represents a significant portion of Apple’s profits, she asked? Cue
said that’s not how Apple looks at it because it doesn’t account for all the work Apple did to make its platform so appealing that an agreement like this could
work as well as it does.

Later, after a closed session in the courtroom and a break for lunch, Google lawyer John Schmidtlein led Cue through a history of the Google / Apple
partnership, and a history of the Safari browser. Cue noted that Safari’s combination of URL and search bar was a user interface innovation, and the seamless
Google integration was part of what made it work. In early promotional materials for Safari, Schmidtlein pointed out, the Google integration was nearly always
mentioned.
</item 3>

<item 4>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.
</item 4>

<item 5>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.
</item 5>

<item 6>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Eddy Cue, in a dark suit, peered down at the monitor in front of him. The screens in the Washington, DC, courtroom had briefly malfunctioned and left witnesses
with only binders, but now the tech was up and running — showing an image of three iPhones, each demonstrating a part of the phone’s setup process. Cue squinted
down at the screen.

“The resolution on this is terrible,” he said. “You should get a Mac.” That got some laughs in an otherwise staid and quiet courtroom. Judge Amit Mehta,
presiding over the case, leaned into his microphone and responded, “If Apple would like to make a donation…” That got even bigger laughs. Then everybody got
back down to business.

Cue was on the stand as a witness in US v. Google, the landmark antitrust trial over Google’s search business.
</item 6>

<item 7>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google CEO Sundar Pichai testified Monday in the U.S. government’s antitrust trial against the company. The executive defended Google’s business tactics,
including its deal with Apple and other partners to make Google the default search engine.

The lawsuit stems from a 2020 antitrust claim over Google’s dominance in the online search market. The claim is separate from one filed in January by the
Department of Justice, along with eight states including New York, California and Colorado, that aims to “halt Google’s anticompetitive scheme, unwind Google’s
monopolistic grip on the market and restore competition to digital advertising.”

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products.

Google argues that it’s just doing business. Everybody wants Google as the default engine because it’s the best. That doesn’t make its actions illegal, the
company says.
</item 7>

<item 8>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 8>

<item 9>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Discussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as
it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.

“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”

Meagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure.
Cue said he’d never really considered that an option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also
argued that the deal was about more than economics and that Apple never seriously considered switching to another provider or building its own search product.
“Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there still isn’t one.
</item 9>

<item 10>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The amount that Google spent on homepage market share has been a fixing point in the trial. Earlier this month, the CEOs of Microsoft and DuckDuckGo testified
that their search engines would have been far more successful, even competitive with Google, had they been able to make similar deals with Apple. Microsoft CEO
Satya Nadella even said he was willing to spend $15 billion per year to get Bing into Apple’s default search, per The Information.

Google agreed not to promote Chrome to Safari users

As part of its search deal with Apple, Google agreed not to promote Chrome to Safari users, reports Bloomberg. Google would have been able to do this with
banners, pop-ups and other annoying means in other Google apps.

The agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their iPhones.

Google tried to be preinstalled on iPhones

Pichai admitted to attempting to get Tim Cook to preinstall Google on every iOS device back in 2018, according to The Verge.
</item 10>
=== END ITEM id=da5a9134df58bbe3 ===

=== ITEM id=2adadb1f48412dd6 ===
Gold sentences and facts:
Sentence 0: “It was the millionaires of the Premier League against the youngsters of the Championship,” reflected the former Denmark forward.
  Fact 0.0: A former Denmark forward reflected on a match.
  Fact 0.1: The former Denmark forward said it was the millionaires of the Premier League against the youngsters of the Championship.
Sentence 1: MLB Network's Jon Morosi reports the superstar is "likely to decide on a team within the next week," according to a source.
  Fact 1.0: MLB Network's Jon Morosi reports that the superstar is likely to decide on a team within the next week.
  Fact 1.1: Jon Morosi's report is according to a source.

Retrieved items:
<item 1>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 1>

<item 2>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Perhaps you are wondering, was the jury also there? Reader, it was.

Bankman-Fried denied he knew about the effectively infinite line of credit Alameda Research received from FTX. This argument was peculiar; essentially my
take-away was that the CEO of a financial company simply didn’t pay attention to finances.

FTX couldn’t get bank accounts right away. Bankman-Fried anticipated it would take a year or two. Rather than wait, he decided to use Alameda as the “payment
provider” for bank transfers. “My understanding at the time was that there were teams managing the process,” he said. “At the time, I wasn’t entirely sure what
was happening.”

Well, sure, understandable! He’s an introvert!

Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the line” to $1 billion

In 2021, FTX was growing to millions of users, with $1 billion of revenue.
</item 2>

<item 3>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 3>

<item 4>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Then, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.

Not having a risk team, when you are any kind of financial anything, is certainly a choice

Though Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly
successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d
fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20
percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.

By the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.
</item 4>

<item 5>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 5>

<item 6>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 6>

<item 7>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 7>

<item 8>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 8>

<item 9>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 9>

<item 10>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 10>
=== END ITEM id=2adadb1f48412dd6 ===

=== ITEM id=128dd9498e94bdd4 ===
Gold sentences and facts:
Sentence 0: Lee was hampered by injury in 2023 but slashed .349/.421/.575 in 2022 with 36 doubles, 10 triples and a career-best 23 home runs.
  Fact 0.0: Lee was hampered by injury in 2023.
  Fact 0.1: Lee slashed .349/.421/.575 in 2022.
  Fact 0.2: Lee hit 36 doubles in 2022.
  Fact 0.3: Lee hit 10 triples in 2022.
  Fact 0.4: Lee hit a career-best 23 home runs in 2022.
Sentence 1: He picked up 16 wickets, conceding just 398 at an economy rate of 4.25 in 10 matches.
  Fact 1.0: Lee picked up 16 wickets in 10 matches.
  Fact 1.1: Lee conceded just 398 in those 10 matches.
  Fact 1.2: Lee's economy rate was 4.25 in those 10 matches.

Retrieved items:

=== END ITEM id=128dd9498e94bdd4 ===
