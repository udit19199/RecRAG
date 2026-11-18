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

This file has 6 items.

=== ITEM id=68cf11efefeac8b5 ===
Gold sentences and facts:
Sentence 0: The video takes place with the band and the album 's two protagonists Christian ( Josh Boswell ) and Gloria ( Lisa Stelly ) taking refuge in a white room after robbing a bank . The police arrive outside the room and open fire through the window , to the couple 's terror . As bullets rain through the room , the band continues playing . Gloria picks up the phone and throws it into a fish tank . As the bullets continue to fly and tear apart the room , Christian and Gloria become calm and walk toward each other , unharmed by the bullets . They embrace and kiss as the room goes dark , recreating the 21st Century Breakdown cover art . As the song ends , the room lights up again , and they are still uninjured . After the room lights up , there are various shots of destruction in the room , including a shot of some of the writing on the walls . The writing includes an excerpt of the lyrics to `` 21 Guns '' , as well as those of the song `` See The Light '' which also appears on `` 21st Century Breakdown '' . The video may represent the meaning of the song in the 21st Century Breakdown story .
  Fact 0.0: The video features the band
  Fact 0.1: The video features the album's two protagonists, Christian and Gloria
  Fact 0.2: Christian is played by Josh Boswell
  Fact 0.3: Gloria is played by Lisa Stelly
  Fact 0.4: Christian and Gloria take refuge in a white room after robbing a bank
  Fact 0.5: The police arrive outside the room and open fire through the window
  Fact 0.6: The police fire to the couple's terror
  Fact 0.7: As bullets rain through the room, the band continues playing
  Fact 0.8: Gloria picks up the phone and throws it into a fish tank
  Fact 0.9: As the bullets continue to fly and tear apart the room, Christian and Gloria become calm
  Fact 0.10: Christian and Gloria walk toward each other, unharmed by the bullets
  Fact 0.11: Christian and Gloria embrace and kiss as the room goes dark
  Fact 0.12: The embrace and kiss recreate the 21st Century Breakdown cover art
  Fact 0.13: As the song ends, the room lights up again
  Fact 0.14: Christian and Gloria are still uninjured when the room lights up
  Fact 0.15: After the room lights up, there are various shots of destruction in the room
  Fact 0.16: The shots include a shot of some of the writing on the walls
  Fact 0.17: The writing includes an excerpt of the lyrics to 21 Guns
  Fact 0.18: The writing also includes lyrics of the song See The Light
  Fact 0.19: The song See The Light also appears on 21st Century Breakdown
  Fact 0.20: The video may represent the meaning of the song in the 21st Century Breakdown story

Retrieved items:
<item 1>
21 Guns (song)
Title Length 1 . `` 21 Guns '' 5 : 21 2 . `` Favorite Son '' ( Green Day 's contribution to the Rock Against Bush Vol. 2 compilation ) 2 : 13 3 . `` 21 Guns ''
( Studio 880 version ) 5 : 17 CD single / 7 '' single No . Title Length 1 . `` 21 Guns '' 5 : 21 2 . `` Favorite Son '' ( From `` Rock Against Bush Vol. 2 '' )
2 : 13 Promo CD No . Title Length 1 . `` 21 Guns '' ( Radio Edit ) 4 : 37 Musical cast version : Digital download / CD single No . Title Length 1 . `` 21 Guns
'' ( Green Day with the cast of American Idiot version ) 4 : 41 21 Guns EP No . Title Length 1 . `` 21 Guns '' ( Green Day with the cast of American Idiot
version ) 4 : 41 2 . `` 21 Guns '' ( Original version ) 5 : 21 3 .
</item 1>

<item 2>
21 Guns (song)
The video was released February 1 . The second version of the song with the cast of American Idiot features Stark Sands , John Gallagher Jr. , and Michael Esper
singing their respective solos instead of Armstrong . It was performed with Armstrong singing lead male vocals at the 52nd Grammy Awards . Live EP 21 Guns Live
EP EP by Green Day Released September 4 , 2009 Recorded July 25 , 2009 in Albany , New York ; July 28 , 2009 Madison Square Garden , New York Genre Pop punk ,
punk rock , alternative rock Length 17 : 52 Label Reprise Producer Green Day Green Day released a live EP for the song including live performances of `` Welcome
to Paradise '' , `` Brain Stew / Jaded '' and `` F.O.D. '' recorded live in Albany , New York and Madison Square Garden . This EP was released on the Australian
iTunes store , Amazon UK , and Napster . Track listing Original version : Digital download / Maxi CD No . Title Length 1 . `` 21 Guns '' 5 : 21 2 .
</item 2>

<item 3>
21 Guns (song)
British Phonographic Industry . Enter 21 Guns in the search field and then press Enter . Jump up ^ `` American single certifications -- Green Day -- 21 Guns ''
. Recording Industry Association of America . Retrieved October 13 , 2015 . If necessary , click Advanced , then click Format , then select Single , then click
SEARCH External links Lyrics of this song at MetroLyrics ( hide ) Green Day Billie Joe Armstrong Mike Dirnt Tré Cool Jason White John Kiffmeyer Studio albums 39
/ Smooth Kerplunk Dookie Insomniac Nimrod Warning American Idiot 21st Century Breakdown ¡ Uno ! ¡ Dos ! ¡ Tré ! Revolution Radio Live albums Live Tracks Tune In
, Tokyo ... Bullet in a Bible Awesome as Fuck Compilation albums 1,039 / Smoothed Out Slappy Hours International Superhits ! Shenanigans Green Day Demolicious
Greatest Hits : God 's Favorite Band EPs 1,000 Hours Slappy Sweet Children Last Night on Earth : Live in Tokyo Singles `` Longview '' `` Welcome to Paradise ''
`` Basket Case '' `` When I Come Around '' `` She '' `` J.A.R. '' `` Geek
</item 3>

<item 4>
21 Guns (song)
rockband.com . Archived from the original on April 25 , 2010 . Retrieved July 12 , 2010 . Jump up ^ `` 21 Guns Live EP '' . GreenDay.com . September 20 , 2009 .
Archived from the original on April 13 , 2010 . Retrieved July 12 , 2010 . Jump up ^ `` 21 Guns Video '' . Green Day.com . June 5 , 2009 . Archived from the
original on June 8 , 2011 . Retrieved July 12 , 2010 . Jump up ^ `` 2009 MTV Video Music Awards Winners '' . MTV . Retrieved July 12 , 2010 . Jump up ^ `` Green
Day -- making of 21 Guns video '' . MySpace Video . June 23 , 2009 . Archived from the original on July 2 , 2009 . Retrieved July 12 , 2010 . Jump up ^ `` Press
Release '' . GreenDay.com . November 13 , 2009 .
</item 4>

<item 5>
21 Guns (song)
21 Guns ( song ) - wikipedia 21 Guns ( song ) Jump to : navigation , search `` 21 Guns '' Single by Green Day from the album 21st Century Breakdown Released May
25 , 2009 ( digital download ) July 14 , 2009 ( CD single ) Format Digital download , CD single , vinyl Recorded 2008 Genre Alternative rock Length 5 : 21 Label
Reprise Songwriter ( s ) Billie Joe Armstrong , David Bowie , Mike Dirnt , and Tré Cool Producer ( s ) Butch Vig and Green Day Green Day singles chronology ``
Know Your Enemy '' ( 2009 ) `` 21 Guns '' ( 2009 ) `` East Jesus Nowhere '' ( 2009 ) `` Know Your Enemy '' ( 2009 ) `` 21 Guns '' ( 2009 ) `` East Jesus Nowhere
'' ( 2009 ) Audio sample file help `` 21 Guns '' is a song by American punk rock band Green Day . It was released as the second single from their eighth album ,
21st Century Breakdown .
</item 5>

<item 6>
21 Guns (song)
The video may represent the meaning of the song in the 21st Century Breakdown story . Musical cast version `` 21 Guns '' Single by Green Day and the cast of
American Idiot from the album American Idiot : The Original Broadway Cast Recording B - side `` Like a Rolling Stone ( Bob Dylan Cover ) '' Released December 3
, 2009 Format Digital download , CD single Recorded 2009 Genre Alternative rock Length 4 : 41 Label Reprise Songwriter ( s ) Billie Joe Armstrong , David Bowie
, Mike Dirnt , and Tré Cool Producer ( s ) Green Day Green Day and the cast of American Idiot singles chronology `` 21 Guns '' ( 2009 ) `` When It 's Time '' (
2010 ) `` 21 Guns ( Musical Cast Version ) '' ( 2010 ) `` When It 's Time '' ( 2010 ) Green Day has recorded two versions of `` 21 Guns '' with the cast of
American Idiot . The recording was produced by Billie Joe Armstrong . The recording with Armstrong singing the male solo parts was released on December 3 to
Spinner.com and also released for digital download on December 22 .
</item 6>

<item 7>
21 Guns (song)
GreenDay.com . November 13 , 2009 . Archived from the original on April 13 , 2010 . Retrieved July 12 , 2010 . Jump up ^ `` Press Release '' . GreenDay.com .
December 2 , 2009 . Archived from the original on June 8 , 2011 . Retrieved July 12 , 2010 . Jump up ^ Officially released video Green Day -- 21 Guns ( Cast
Version ) on YouTube Jump up ^ `` Get The New 21 Guns EP In Itunes Now '' . GreenDay.com . February 2 , 2010 . Archived from the original on June 8 , 2011 .
Retrieved July 12 , 2010 . Jump up ^ `` Australian-charts.com -- Green Day -- 21 Guns '' . ARIA Top 50 Singles . Retrieved October 27 , 2017 . Jump up ^ ``
Austriancharts.at -- Green Day -- 21 Guns '' ( in German ) . Ö3 Austria Top 40 . Retrieved October 27 , 2017 .
</item 7>

<item 8>
21 Guns (song)
Retrieved June 16 , 2012 . Jump up ^ `` Green Day -- 21 Guns ( CD ) at Discogs '' . Discogs . Retrieved June 16 , 2012 . Jump up ^ `` Transformers : Revenge of
the Fallen Soundtrack '' . digiRAMA . Archived from the original on June 29 , 2009 . Retrieved July 12 , 2010 . Jump up ^ `` MTV Games , Harmonix & Warner Bros.
/ Reprise Records Partner With Green Day to Bring Their Music Exclusively to Rock Band / / News / / Rock Band '' . Rockband.com . Retrieved August 21 , 2009 .
Jump up ^ `` SkinnyLove21 's Journal -- The Vampire Diaries Soundtrack , Season 1 '' . last.fm. April 16 , 2010 . Archived from the original on August 18 , 2010
. Retrieved July 12 , 2010 . Jump up ^ `` '' 21 Guns '' -- Green Day `` . rockband.com . Archived from the original on April 25 , 2010 .
</item 8>

<item 9>
21 Guns (song)
Top 40 Singles . Retrieved October 27 , 2017 . Jump up ^ `` Green Day -- 21 Guns . Polish Singles Chart '' . Jump up ^ `` Spanishcharts.com -- Green Day -- 21
Guns '' Canciones Top 50 . Retrieved October 27 , 2017 . Jump up ^ `` Swedishcharts.com -- Green Day -- 21 Guns '' . Singles Top 100 . Retrieved October 27 ,
2017 . Jump up ^ `` Swisscharts.com -- Green Day -- 21 Guns '' . Swiss Singles Chart . Retrieved October 27 , 2017 . Jump up ^ `` Green Day : Artist Chart
History '' . Official Charts Company . Retrieved October 27 , 2017 . Jump up ^ `` Green Day Chart History ( Hot 100 ) '' . Billboard . Retrieved October 27 ,
2017 . Jump up ^ `` Green Day Chart History ( Pop Songs ) '' . Billboard . Retrieved October 27 , 2017 . Jump up ^ `` Green Day Chart History ( Adult Pop Songs
) '' . Billboard .
</item 9>

<item 10>
21 Guns (song)
It became available for download on the music video game series Rock Band on July 7 , 2009 , along with the songs `` East Jesus Nowhere '' and `` Know Your
Enemy . The song was featured in episode 6 of The Vampire Diaries . The song is featured in the video game Green Day : Rock Band and is downloadable content for
the Rock Band series . A live version was also released on 21 Guns Live EP in September 2009 . Music video A music video was directed by Marc Webb and filmed in
Los Angeles on June 6 , 2009 . It premiered on MySpace on Monday , June 22 , 2009 at midnight EDT . A shorter version of the video also exists , which is set to
the radio edit of the song . It is the third Green Day video to feature touring guitarist Jason White performing with the band , following `` Wake Me Up When
September Ends '' and `` Working Class Hero '' . He is also seen briefly in the music videos for `` When I Come Around '' and `` Jesus of Suburbia '' but he is
not performing with the band .
</item 10>
=== END ITEM id=68cf11efefeac8b5 ===

=== ITEM id=39e0959eec4988d2 ===
Gold sentences and facts:
Sentence 0: Fortnite is set in contemporary Earth , where the sudden appearance of a worldwide storm causes 98 % of the world 's population to disappear , and zombie - like creatures rise to attack the remainder . Considered by Epic as a cross between Minecraft and Left 4 Dead , Fortnite has up to four players cooperating on various missions on randomly - generated maps to collect resources , build fortifications around defensive objectives that are meant to help fight the storm and protect survivors , and construct weapons and traps to engage in combat with waves of these creatures that attempt to destroy the objectives . Players gain rewards through these missions to improve their hero characters , support teams , and arsenal of weapon and trap schematics to be able to take on more difficult missions . The game is supported through microtransactions to purchase in - game currency that can be used towards these upgrades .
  Fact 0.0: Fortnite is set in contemporary Earth.
  Fact 0.1: A sudden appearance of a worldwide storm causes 98% of the world's population to disappear.
  Fact 0.2: Zombie-like creatures rise to attack the remainder of the population.
  Fact 0.3: Epic considers Fortnite a cross between Minecraft and Left 4 Dead.
  Fact 0.4: Fortnite has up to four players cooperating on various missions.
  Fact 0.5: The missions are on randomly generated maps.
  Fact 0.6: Players collect resources in the missions.
  Fact 0.7: Players build fortifications around defensive objectives.
  Fact 0.8: The fortifications are meant to help fight the storm and protect survivors.
  Fact 0.9: Players construct weapons and traps.
  Fact 0.10: Players engage in combat with waves of zombie-like creatures.
  Fact 0.11: The creatures attempt to destroy the objectives.
  Fact 0.12: Players gain rewards through the missions.
  Fact 0.13: Rewards are used to improve hero characters, support teams, and arsenal of weapon and trap schematics.
  Fact 0.14: The rewards help players take on more difficult missions.
  Fact 0.15: The game is supported through microtransactions.
  Fact 0.16: Microtransactions purchase in-game currency.
  Fact 0.17: In-game currency can be used towards these upgrades.
Sentence 1: Fortnite was first revealed at the 2011 Spike Video Game Awards ( VGA ) , with Epic 's former design director Cliff Bleszinski introducing a trailer for the game . Donald Mustard , creative lead at Epic , said in 2017 that this announcement was `` three weeks after we came up with the idea , before we even made the game '' . The title , which started out as an internal game jam project , represents a departure from the company 's previous work . As Bleszinski explained during the Spike event , Epic wanted to `` switch things up a little bit and do something different and fun '' with Fortnite , describing it as `` a world where you explore , you scavenge , you build and ultimately you survive . '' In an interview with Engadget , he also echoed these statements , claiming that the game would be different from the Gears of War series : `` There 's no dudebros in it ... Not that there 's anything wrong with that , right ? But creatively for the team , Gears has been amazing for us . But it 's fun to kind of stretch our wings and do something that 's a little different from the usual . '' Epic Games Vice President of Publishing Mike Fischer said in 2015 that Epic recognized that they `` announced this game too soon '' , and that its lengthy development period was due to `` very good reasons . ''
  Fact 1.0: Fortnite was first revealed at the 2011 Spike Video Game Awards (VGA).
  Fact 1.1: Epic's former design director Cliff Bleszinski introduced a trailer for the game at the 2011 Spike Video Game Awards.
  Fact 1.2: Donald Mustard, creative lead at Epic, said in 2017 that the announcement was three weeks after we came up with the idea, before we even made the game.
  Fact 1.3: Fortnite started out as an internal game jam project.
  Fact 1.4: Fortnite represents a departure from the company's previous work.
  Fact 1.5: Bleszinski said Epic wanted to switch things up a little bit and do something different and fun with Fortnite.
  Fact 1.6: Bleszinski described Fortnite as a world where you explore, you scavenge, you build and ultimately you survive.
  Fact 1.7: In an interview with Engadget, Bleszinski claimed the game would be different from the Gears of War series.
  Fact 1.8: Bleszinski said there are no dudebros in Fortnite.
  Fact 1.9: Bleszinski said Gears has been amazing for the team.
  Fact 1.10: Bleszinski said it is fun to stretch the team's wings and do something a little different from the usual.
  Fact 1.11: Epic Games Vice President of Publishing Mike Fischer said in 2015 that Epic recognized they announced this game too soon.
  Fact 1.12: Mike Fischer said the lengthy development period was due to very good reasons.

Retrieved items:
<item 1>
Fortnite
Fortnite was being developed alongside Paragon , which Epic announced on November 2015 . As Paragon seemed to take Epic 's focus , leaving little news about
Fortnite , CEO Tim Sweeney said in March 2016 that they were still committed to Fortnite once Paragon was launched and established , given that much of the work
on Fortnite would take time to get the right balance for gameplay . `` We figure we should start with one major successful launch and do one at a time .
Fortnite will be next . '' By June 2017 , Epic Games announced that Fortnite was now set for a 2018 release across Windows , macOS , and the PlayStation 4 and
Xbox One consoles . Leading up to this free - to - play release , the game was offered as a paid early access period starting on July 25 , 2017 for all
platforms ; players who pre-ordered Founder 's Packs were granted access to the game on July 21 . The lengthy period since the game 's alpha phases was ascribed
to developing Fortnite as a games - as - a-service model , according to creative lead Donald Mustard .
</item 1>

<item 2>
Fortnite
They further opted for personal computer exclusivity to avoid the difficulty of having to go through console certification , and as they planned to be
constantly monitoring and tweaking the game , acting as a dungeon master , the personal computer approach would allow them to do this without restrictions
normally set by console manfuctures . Bleszinski later clarified that they would not rule out release on other platforms as they developed the title . Fortnite
cosplayers at Gamescom 2017 Fortnite 's development was spread among several of Epic 's satellite studios , and was also co-developed by the Polish studio
People Can Fly , which had worked with Epic previously on earlier games , and had been fully acquired by Epic sometime in 2012 . People Can Fly were briefly
renamed Epic Games Poland in 2013 as to align with Epic 's other studios . By March 2014 , there were about 90 developers working on the game . People Can Fly
later returned to being an independent studio and their own name in 2015 , but continued to help Epic with Fortnite 's development . During Fortnite 's
development , Epic had seen the game industry shifting to games as a service model .
</item 2>

<item 3>
Fortnite
Jump up ^ Brown , Fraser ( December 10 , 2014 ) . `` Epic pulls back the curtain ( and flesh ) on Fortnite 's monster design '' . PCGamesN . Archived from the
original on July 30 , 2017 . Retrieved July 29 , 2017 . Jump up ^ Greening , Chris ( October 22 , 2016 ) . `` Game Releases Calendar : 2016 Edition '' . Game
Music Online . Archived from the original on September 30 , 2017 . Retrieved July 16 , 2017 . Jump up ^ McWhertor , Michael ( July 12 , 2012 ) . `` Epic Games '
' Fortnite ' will be the developer 's first Unreal Engine 4 game '' . Polygon . Archived from the original on July 30 , 2017 . Retrieved August 1 , 2017 . Jump
up ^ Makuch , Eddie ( July 12 , 2013 ) . `` Epic : Fortnite is `` Minecraft meets Left 4 Dead '' `` . GameSpot . Archived from the original on March 27 , 2015 .
</item 3>

<item 4>
Fortnite
Fortnite - wikipedia Fortnite Jump to : navigation , search For the battle royale version of the game , see Fortnite Battle Royale . Fortnite European box art
Developer ( s ) Epic Games People Can Fly Publisher ( s ) Epic Games Director ( s ) Darren Sugg Artist ( s ) Pete Ellis Composer ( s ) Rom Di Prisco Engine
Unreal Engine 4 Platform ( s ) Microsoft Windows macOS PlayStation 4 Xbox One iOS Release 2018 Genre ( s ) Survival Mode ( s ) Single - player , multiplayer
Fortnite is a co-op sandbox survival game developed by Epic Games and People Can Fly and published by Epic Games . The game was released as a paid - for early
access title for Microsoft Windows , macOS , PlayStation 4 and Xbox One on July 25 , 2017 , with a full free - to - play release expected in 2018 . The retail
versions of the game were published by Gearbox Publishing , while online distribution of the PC versions is handled by Epic 's launcher . Fortnite is set in
contemporary Earth , where the sudden appearance of a worldwide storm causes 98 % of the world 's population to disappear , and zombie - like creatures rise to
attack the remainder .
</item 4>

<item 5>
Fortnite
Considered by Epic as a cross between Minecraft and Left 4 Dead , Fortnite has up to four players cooperating on various missions on randomly - generated maps
to collect resources , build fortifications around defensive objectives that are meant to help fight the storm and protect survivors , and construct weapons and
traps to engage in combat with waves of these creatures that attempt to destroy the objectives . Players gain rewards through these missions to improve their
hero characters , support teams , and arsenal of weapon and trap schematics to be able to take on more difficult missions . The game is supported through
microtransactions to purchase in - game currency that can be used towards these upgrades . A standalone mode , Fortnite Battle Royale , based on the battle
royale game genre but based on the core Fortnite gameplay , was released for the same platforms in September 2017 .
</item 5>

<item 6>
Fortnite
Engadget . Archived from the original on October 1 , 2017 . Retrieved July 24 , 2017 . Jump up ^ Orland , Kyle ( June 8 , 2015 ) . `` Hands - on : Fortnite is
an overwhelming zombie defense experience '' . Ars Technica . Archived from the original on December 8 , 2017 . Retrieved December 7 , 2017 . Jump up ^ Yin -
Pool , Wesley ( December 12 , 2011 ) . `` Epic reveals Minecraft inspiration for Fortnite '' . Eurogamer . Archived from the original on December 1 , 2017 .
Retrieved November 29 , 2017 . Jump up ^ Rigney , Ryan ( July 12 , 2012 ) . `` First Screens : Epic 's Fortnite Is the First Unreal Engine 4 Game '' . Wired .
Archived from the original on December 22 , 2016 . Retrieved November 29 , 2017 . Jump up ^ Hafer , T.J. ( July 12 , 2012 ) .
</item 6>

<item 7>
Fortnite
Development History Fortnite was first revealed at the 2011 Spike Video Game Awards ( VGA ) , with Epic 's former design director Cliff Bleszinski introducing a
trailer for the game . Donald Mustard , creative lead at Epic , said in 2017 that this announcement was `` three weeks after we came up with the idea , before
we even made the game '' . The title , which started out as an internal game jam project , represents a departure from the company 's previous work . As
Bleszinski explained during the Spike event , Epic wanted to `` switch things up a little bit and do something different and fun '' with Fortnite , describing
it as `` a world where you explore , you scavenge , you build and ultimately you survive . '' In an interview with Engadget , he also echoed these statements ,
claiming that the game would be different from the Gears of War series : `` There 's no dudebros in it ... Not that there 's anything wrong with that , right ?
But creatively for the team , Gears has been amazing for us . But it 's fun to kind of stretch our wings and do something that 's a little different from the
usual . ''
</item 7>

<item 8>
Fortnite
Contents ( hide ) 1 Synopsis 2 Gameplay 3 Development 3.1 History 3.2 Art and design 4 Reception 4.1 Sales 4.2 Accolades 5 References 6 External links Synopsis
One day , 98 % of Earth 's population suddenly disappeared , and the remaining population found the skies covered in dense clouds , creating chaotic storms that
dropped husks , humanoid zombie - like creatures , that attacked the living . The survivors found ways to construct `` storm shields '' , a field that cleared
the storm clouds from immediately overhead and reduced the attacks from husks , and used these to set up survivor bases across the globe . The player is a
commander of one of these bases , charged with going out of the storm shield to find resources , survivors , and other allies to help expand their storm shield
and find a way to return Earth to its normal state . Gameplay Currently , the paid - for product Fortnite provides two distinct modes : access to Fortnite
Battle Royale , ( which is also available as a separate free - to - play title ) and the cooperative player - versus - environment `` Save the World '' , which
is unique to the main Fortnite game .
</item 8>

<item 9>
Fortnite
( July 12 , 2012 ) . `` Fortnite will be a PC exclusive and Epic 's first Unreal Engine 4 game '' . PC Gamer . Archived from the original on December 1 , 2017 .
Retrieved November 29 , 2017 . Jump up ^ McGee , Maxwell ( July 13 , 2012 ) . `` Epic 's Cliff Bleszinski and Tanya Jessen Talk Up Fortnite '' . GameSpot .
Archived from the original on February 6 , 2018 . Retrieved November 29 , 2017 . ^ Jump up to : Purchase , Robert ( September 1 , 2012 ) . `` Fortnite detailed
at PAX Prime ; Minecraft meets DayZ '' . Eurogamer . Archived from the original on December 1 , 2017 . Retrieved November 29 , 2017 . Jump up ^ Sheridan ,
Connor ( November 1 , 2013 ) . `` People Can Fly working on Fortnite , renamed Epic Games Poland '' . GamesRadar . Archived from the original on December 2 ,
2017 .
</item 9>

<item 10>
Fortnite
Epic Games Vice President of Publishing Mike Fischer said in 2015 that Epic recognized that they `` announced this game too soon '' , and that its lengthy
development period was due to `` very good reasons . '' As the game was at its very preliminary stages at the VGA reveal , the goal of this reveal was to seek
public interest in the title and potential publishing partners as to decide on the game 's release platforms and timeframe . During the July 2012 San Diego
Comic Con , Epic announced that Fortnite would be an exclusive personal computer title , and the first one to be developed by Epic using their new Unreal 4 game
engine , with a planned release in 2013 . The game 's development was originally started in the Unreal 3 engine , but as they progressed , they had seen the
opportunity to work in several of the new feature sets and scripting language offered by Unreal 4 for Fortnite , while still running on most personal computers
at that time .
</item 10>
=== END ITEM id=39e0959eec4988d2 ===

=== ITEM id=a3cdb4c68e75a12d ===
Gold sentences and facts:
Sentence 0: The formation of ATP from ADP and P is energetically unfavorable and would normally proceed in the reverse direction . In order to drive this reaction forward , ATPase couples ATP synthesis during cellular respiration to an electrochemical gradient created by the difference in proton ( H ) concentration across the mitochondrial membrane in eukaryotes or the plasma membrane in bacteria . During photosynthesis in plants , ATP is synthesized by ATPase using a proton gradient created in the thylakoid lumen through the thylakoid membrane and into the chloroplast stroma .
  Fact 0.0: The formation of ATP from ADP and P is energetically unfavorable.
  Fact 0.1: The formation of ATP from ADP and P would normally proceed in the reverse direction.
  Fact 0.2: ATPase couples ATP synthesis during cellular respiration to an electrochemical gradient in order to drive the reaction forward.
  Fact 0.3: The electrochemical gradient is created by the difference in proton (H) concentration across the mitochondrial membrane in eukaryotes.
  Fact 0.4: The electrochemical gradient is created by the difference in proton (H) concentration across the plasma membrane in bacteria.
  Fact 0.5: During photosynthesis in plants, ATP is synthesized by ATPase using a proton gradient.
  Fact 0.6: During photosynthesis in plants, the proton gradient is created in the thylakoid lumen through the thylakoid membrane.
  Fact 0.7: During photosynthesis in plants, the proton gradient passes into the chloroplast stroma.
Sentence 1: Located within the thylakoid membrane and the inner mitochondrial membrane , ATP synthase consists of two regions F and F. F causes rotation of F and is made of c - ring and subunits a , b , d , F6. F is made of α , β , γ , δ ( \ displaystyle \ alpha , \ beta , \ gamma , \ delta ) subunits . F has a water soluble par that can hydrolyze ATP . F on the other hand has mainly hydrophobic regions . F F creates a pathway for protons movement across the membrane .
  Fact 1.0: ATP synthase is located within the thylakoid membrane and the inner mitochondrial membrane.
  Fact 1.1: ATP synthase consists of two regions, F and F.
  Fact 1.2: The F region causes rotation of the F region.
  Fact 1.3: The F region is made of a c-ring and subunits a, b, d, and F6.
  Fact 1.4: The F region is made of alpha, beta, gamma, and delta subunits.
  Fact 1.5: The F region has a water soluble part that can hydrolyze ATP.
  Fact 1.6: The F region has mainly hydrophobic regions.
  Fact 1.7: The F and F regions together create a pathway for protons to move across the membrane.
Sentence 2: The crystal structure of the F showed alternating alpha and beta subunits ( 3 of each ) , arranged like segments of an orange around a rotating asymmetrical gamma subunit . According to the current model of ATP synthesis ( known as the alternating catalytic model ) , the transmembrane potential created by ( H+ ) proton cations supplied by the electron transport chain , drives the ( H+ ) proton cations from the intermembrane space through the membrane via the F region of ATP synthase . A portion of the F ( the ring of c - subunits ) rotates as the protons pass through the membrane . The c - ring is tightly attached to the asymmetric central stalk ( consisting primarily of the gamma subunit ) , causing it to rotate within the alpha beta of F causing the 3 catalytic nucleotide binding sites to go through a series of conformational changes that leads to ATP synthesis . The major F subunits are prevented from rotating in sympathy with the central stalk rotor by a peripheral stalk that joins the alpha beta to the non-rotating portion of F. The structure of the intact ATP synthase is currently known at low - resolution from electron cryo - microscopy ( cryo - EM ) studies of the complex . The cryo - EM model of ATP synthase suggests that the peripheral stalk is a flexible structure that wraps around the complex as it joins F to F. Under the right conditions , the enzyme reaction can also be carried out in reverse , with ATP hydrolysis driving proton pumping across the membrane .
  Fact 2.0: The crystal structure of the F region showed alternating alpha and beta subunits, three of each.
  Fact 2.1: The alpha and beta subunits are arranged like segments of an orange around a rotating asymmetrical gamma subunit.
  Fact 2.2: The current model of ATP synthesis is known as the alternating catalytic model.
  Fact 2.3: In the alternating catalytic model, the transmembrane potential created by proton cations supplied by the electron transport chain drives proton cations from the intermembrane space through the membrane via the F region of ATP synthase.
  Fact 2.4: A portion of the F region, the ring of c-subunits, rotates as protons pass through the membrane.
  Fact 2.5: The c-ring is tightly attached to the asymmetric central stalk, which consists primarily of the gamma subunit.
  Fact 2.6: The c-ring causes the central stalk to rotate within the alpha and beta subunits of F.
  Fact 2.7: The rotation causes the 3 catalytic nucleotide binding sites to go through a series of conformational changes that leads to ATP synthesis.
  Fact 2.8: The major F subunits are prevented from rotating in sympathy with the central stalk rotor by a peripheral stalk.
  Fact 2.9: The peripheral stalk joins the alpha and beta subunits to the non-rotating portion of F.
  Fact 2.10: The structure of the intact ATP synthase is currently known at low resolution from electron cryo-microscopy (cryo-EM) studies of the complex.
  Fact 2.11: The cryo-EM model of ATP synthase suggests that the peripheral stalk is a flexible structure that wraps around the complex as it joins F to F.
  Fact 2.12: Under the right conditions, the enzyme reaction can also be carried out in reverse.
  Fact 2.13: In reverse, ATP hydrolysis drives proton pumping across the membrane.

Retrieved items:
<item 1>
Cellular respiration
Aerobic respiration summed up 1 . Glycolysis : -- - 2 ATPs + Glucose → 2 Pyruvic Acid + 4 Hydrogen + 4 ATPs 2 . Formation of Acetyl CoA : -- - 2 Pyruvic Acid +
2 CoA → 2 Acetyl CoA + 2 Carbon Dioxide + 2 Hydrogen 3 . Krebs Cycle : -- - 2 Acetyl CoA + 3 O2 → 6 Hydrogen + 4 Carbon Dioxide + 2 ATPs 4 . Electron Transport
System : -- - 12 Hydrogen + 3 O2 → 6 Water + 32 ATPs Overall Reaction : -- - Glucose + 6 O2 → 6 Carbon Dioxide + 6 Water + 36 ATPs Glycolysis Out of the
cytoplasm it goes into the Krebs cycle with the acetyl CoA . It then mixes with CO and makes 2 ATP , NADH , and FADH . From there the NADH and FADH go into the
NADH reductase , which produces the enzyme .
</item 1>

<item 2>
Cellular respiration
The electrons are finally transferred to exogenous oxygen and , with the addition of two protons , water is formed . Efficiency of ATP production The table
below describes the reactions involved when one glucose molecule is fully oxidized into carbon dioxide . It is assumed that all the reduced coenzymes are
oxidized by the electron transport chain and used for oxidative phosphorylation . Step coenzyme yield ATP yield Source of ATP Glycolysis preparatory phase − 2
Phosphorylation of glucose and fructose 6 - phosphate uses two ATP from the cytoplasm .
</item 2>

<item 3>
ATP synthase
According to the current model of ATP synthesis ( known as the alternating catalytic model ) , the transmembrane potential created by ( H+ ) proton cations
supplied by the electron transport chain , drives the ( H+ ) proton cations from the intermembrane space through the membrane via the F region of ATP synthase .
A portion of the F ( the ring of c - subunits ) rotates as the protons pass through the membrane . The c - ring is tightly attached to the asymmetric central
stalk ( consisting primarily of the gamma subunit ) , causing it to rotate within the alpha beta of F causing the 3 catalytic nucleotide binding sites to go
through a series of conformational changes that leads to ATP synthesis . The major F subunits are prevented from rotating in sympathy with the central stalk
rotor by a peripheral stalk that joins the alpha beta to the non-rotating portion of F. The structure of the intact ATP synthase is currently known at low -
resolution from electron cryo - microscopy ( cryo - EM ) studies of the complex .
</item 3>

<item 4>
Cellular respiration
The NADH pulls the enzyme 's electrons to send through the electron transport chain . The electron transport chain pulls H ions through the chain . From the
electron transport chain , the released hydrogen ions make ADP for an end result of 32 ATP . O attracts itself to the left over electron to make water . Lastly
, ATP leaves through the ATP channel and out of the mitochondria . Main article : Glycolysis Glycolysis is a metabolic pathway that takes place in the cytosol
of cells in all living organisms . This pathway can function with or without the presence of oxygen . In humans , aerobic conditions produce pyruvate and
anaerobic conditions produce lactate . In aerobic conditions , the process converts one molecule of glucose into two molecules of pyruvate ( pyruvic acid ) ,
generating energy in the form of two net molecules of ATP . Four molecules of ATP per glucose are actually produced , however , two are consumed as part of the
preparatory phase . The initial phosphorylation of glucose is required to increase the reactivity ( decrease its stability ) in order for the molecule to be
cleaved into two pyruvate molecules by the enzyme aldolase .
</item 4>

<item 5>
ATP synthase
Depiction of ATP synthase using the chemiosmotic proton gradient to power ATP synthesis through oxidative phosphorylation . In the 1960s through the 1970s ,
Paul Boyer , a UCLA Professor , developed the binding change , or flip - flop , mechanism theory , which postulated that ATP synthesis is dependent on a
conformational change in ATP synthase generated by rotation of the gamma subunit . The research group of John E. Walker , then at the MRC Laboratory of
Molecular Biology in Cambridge , crystallized the F catalytic - domain of ATP synthase . The structure , at the time the largest asymmetric protein structure
known , indicated that Boyer 's rotary - catalysis model was , in essence , correct . For elucidating this , Boyer and Walker shared half of the 1997 Nobel
Prize in Chemistry . The crystal structure of the F showed alternating alpha and beta subunits ( 3 of each ) , arranged like segments of an orange around a
rotating asymmetrical gamma subunit .
</item 5>

<item 6>
ATP synthase
Large - enough quantities of ATP cause it to create a transmembrane proton gradient , this is used by fermenting bacteria that do not have an electron transport
chain , but rather hydrolyze ATP to make a proton gradient , which they use to drive flagella and the transport of nutrients into the cell . In respiring
bacteria under physiological conditions , ATP synthase , in general , runs in the opposite direction , creating ATP while using the proton motive force created
by the electron transport chain as a source of energy . The overall process of creating energy in this fashion is termed oxidative phosphorylation . The same
process takes place in the mitochondria , where ATP synthase is located in the inner mitochondrial membrane and the F - part projects into mitochondrial matrix
. The consumption of ATP by ATP - synthase pumps proton cations into the matrix . Evolution ( edit ) The evolution of ATP synthase is thought to have been
modular whereby two functionally independent subunits became associated and gained new functionality . This association appears to have occurred early in
evolutionary history , because essentially the same structure and activity of ATP synthase enzymes are present in all kingdoms of life .
</item 6>

<item 7>
ATP synthase
ATP synthase - wikipedia ATP synthase Jump to : navigation , search ATP Synthase Molecular model of ATP synthase determined by X-ray crystallography . Stator is
not shown here . Identifiers EC number 3.6. 3.14 CAS number 9000 - 83 - 3 Databases IntEnz IntEnz view BRENDA BRENDA entry ExPASy NiceZyme view KEGG KEGG entry
MetaCyc metabolic pathway PRIAM profile PDB structures RCSB PDB PDBe PDBsum Gene Ontology AmiGO / QuickGO ( show ) Search PMC articles PubMed articles NCBI
proteins ATP synthase is an enzyme that creates the energy storage molecule adenosine triphosphate ( ATP ) . ATP is the most commonly used `` energy currency ''
of cells for most organisms . It is formed from adenosine diphosphate ( ADP ) and inorganic phosphate ( P ) .
</item 7>

<item 8>
ATP synthase
The overall reaction catalyzed by ATP synthase is : ADP + P + H ⇌ ATP + H O + H The formation of ATP from ADP and P is energetically unfavorable and would
normally proceed in the reverse direction . In order to drive this reaction forward , ATPase couples ATP synthesis during cellular respiration to an
electrochemical gradient created by the difference in proton ( H ) concentration across the mitochondrial membrane in eukaryotes or the plasma membrane in
bacteria . During photosynthesis in plants , ATP is synthesized by ATPase using a proton gradient created in the thylakoid lumen through the thylakoid membrane
and into the chloroplast stroma . ATP synthase consists of two main subunits , F and F , which has a rotational motor mechanism allowing for ATP production .
Because of its rotating subunit ATP Synthase is a molecular machine .
</item 8>

<item 9>
ATP synthase
This may have evolved to carry out the reverse reaction and act as an ATP synthase . In different species ( edit ) E coli ( edit ) E. coli ATP synthase is the
simplest known form of ATP synthase , with 8 different subunit types . Yeast ( edit ) Yeast ATP synthase is one of the best - studied eukaryotic ATP synthases ;
and five F , eight F subunits , and seven associated proteins have been identified . Most of these proteins have homologues in other eukaryotes . Plant ( edit )
In plants , ATP synthase is also present in chloroplasts ( CF F - ATP synthase ) . The enzyme is integrated into thylakoid membrane ; the CF - part sticks into
stroma , where dark reactions of photosynthesis ( also called the light - independent reactions or the Calvin cycle ) and ATP synthesis take place . The overall
structure and the catalytic mechanism of the chloroplast ATP synthase are almost the same as those of the mitochondrial enzyme . However , in chloroplasts , the
proton motive force is generated not by respiratory electron transport chain but by primary photosynthetic proteins .
</item 9>

<item 10>
Cellular respiration
Isocitrate is modified to become α - ketoglutarate ( 5 carbons ) , succinyl - CoA , succinate , fumarate , malate , and , finally , oxaloacetate . The net gain
of high - energy compounds from one cycle is 3 NADH , 1 FADH , and 1 GTP ; the GTP may subsequently be used to produce ATP . Thus , the total yield from 1
glucose molecule ( 2 pyruvate molecules ) is 6 NADH , 2 FADH , and 2 ATP . Oxidative phosphorylation Main articles : Oxidative phosphorylation , Electron
transport chain , Electrochemical gradient , and ATP synthase In eukaryotes , oxidative phosphorylation occurs in the mitochondrial cristae . It comprises the
electron transport chain that establishes a proton gradient ( chemiosmotic potential ) across the boundary of inner membrane by oxidizing the NADH produced from
the Krebs cycle . ATP is synthesized by the ATP synthase enzyme when the chemiosmotic gradient is used to drive the phosphorylation of ADP .
</item 10>
=== END ITEM id=a3cdb4c68e75a12d ===

=== ITEM id=bfa9172e1daaecc3 ===
Gold sentences and facts:
Sentence 0: Compared to the acetabulum ( at the hip - joint ) the glenoid cavity is relatively shallow . This makes the shoulder joint prone to dislocation ( luxation ) . Strong glenohumeral ligaments and muscles prevents dislocation in most cases .
  Fact 0.0: The glenoid cavity is relatively shallow compared to the acetabulum, which is at the hip joint.
  Fact 0.1: The shallow glenoid cavity makes the shoulder joint prone to dislocation (luxation).
  Fact 0.2: Strong glenohumeral ligaments and muscles prevent dislocation in most cases.

Retrieved items:
<item 1>
Glenoid cavity
Glenoid cavity - wikipedia Glenoid cavity Jump to : navigation , search Glenoid cavity Costal surface of left scapula . Glenoid cavity shown in red . Glenoid
cavity shown in red . This cavity articulates with the head of the humerus . Details Identifiers Latin cavitas glenoidalis , fossa glenoidalis Dorlands /
Elsevier c_16 / 12220465 TA A02. 4.01. 019 FMA 23275 Anatomical terms of bone ( edit on Wikidata ) The glenoid cavity or glenoid fossa of scapula is a part of
the shoulder . It is a shallow , pyriform articular surface , which is located on the lateral angle of the scapula . It is directed laterally and forward and
articulates with the head of the humerus ; it is broader below than above and its vertical diameter is the longest . This cavity forms the glenohumeral joint
along with the humerus . This type of joint is classified as a synovial , ball and socket joint .
</item 1>

<item 2>
Glenoid cavity
This type of joint is classified as a synovial , ball and socket joint . The humerus is held in place within the glenoid cavity by means of the long head of the
biceps tendon . This tendon originates on the superior margin of the glenoid cavity and loops over the shoulder , bracing humerus against the cavity . The
rotator cuff also reinforces this joint more specifically with the supraspinatus tendon to hold the head of the humerus in the glenoid cavity . The cavity
surface is covered with cartilage in the fresh state ; and its margins , slightly raised , give attachment to a fibrocartilaginous structure , the glenoid
labrum , which deepens the cavity . This cartilage is very susceptible to tearing . When torn , it is most commonly known as a SLAP lesion which is generally
caused by repetitive shoulder movements . Compared to the acetabulum ( at the hip - joint ) the glenoid cavity is relatively shallow . This makes the shoulder
joint prone to dislocation ( luxation ) . Strong glenohumeral ligaments and muscles prevents dislocation in most cases .
</item 2>

<item 3>
Glenoid cavity
at 16 : 19 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the
Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 3>

<item 4>
Glenoid cavity
ISBN 1 - 4051 - 3413 - 5 . This article incorporates text in the public domain from the 20th edition of Gray 's Anatomy ( 1918 ) References ( edit ) Larson ,
Susan G. ( 2009 ) . `` Evolution of the Hominin Shoulder : Early Homo '' . In Grine , Frederick E. ; Fleagle , John G. ; Leakey , Richard E. The First Humans --
Origin and Early Evolution of the Genus Homo . Springer . doi : 10.1007 / 978 - 1 - 4020 - 9980 - 9 . ISBN 978 - 1 - 4020 - 9979 - 3 . ANATOMY & PHYSIOLOGY :
THE UNITY OF FORM AND FUNCTION , SIXTH EDITION Published by McGraw - Hill Written by Kenneth Saladin
http://www.ucsfhealth.org/conditions/glenoid_labrum_tear/index.html External links ( edit ) Wikimedia Commons has media related to Glenoid cavity . Diagram at
cerrocoso.edu Anatomy figure : 03 : 02 - 07 at Human Anatomy Online ,
</item 4>

<item 5>
Glenoid cavity
Animation of glenohumeral joint . The muscle shown is supraspinatus muscle . See also ( edit ) Glenopolar angle Notes ( edit ) Jump up ^ The word glenoid is
pronounced / ˈɡliːnɔɪd / or / ˈɡlɛnɔɪd / ( both are common ) and is from Greek : gléne , `` socket '' , reflecting the shoulder joint 's ball - and - socket
form . References ( edit ) Jump up ^ `` Scapulohumeral Rhythm - Physiopedia , universal access to physiotherapy knowledge '' . www.physio-pedia.com . Retrieved
2016 - 09 - 01 . Jump up ^ Larson 2009 , p. 65 Jump up ^ Martin , A.J. ( 2006 ) . Introduction to the Study of Dinosaurs . Second Edition . Oxford , Blackwell
Publishing . pg. 299 - 300 . ISBN 1 - 4051 - 3413 - 5 .
</item 5>

<item 6>
Glenoid cavity
edu Anatomy figure : 03 : 02 - 07 at Human Anatomy Online , SUNY Downstate Medical Center Mechanics of Glenohumeral Instability at University of Washington
Department of Orthopaedics ( hide ) Bones of the arm Shoulder girdle , clavicle conoid tubercle trapezoid line costal tuberosity subclavian groove Scapula
fossae ( subscapular , supraspinatous , infraspinatous ) notches ( suprascapular , great scapular ) glenoid cavity tubercles ( infraglenoid , supraglenoid )
spine of scapula acromion coracoid process angles ( superior , inferior , lateral ) Humerus upper extremity : necks ( anatomical , surgical ) tubercles (
greater , lesser ) bicipital groove body : radial sulcus deltoid tuberosity lower extremity : capitulum trochlea epicondyles ( lateral , medial ) supracondylar
ridges ( lateral , medial ) fossae ( radial , coronoid , olecranon ) Forearm Radius near elbow ( head , tuberosity ) near wrist ( ulnar notch , styloid process
,
</item 6>

<item 7>
Glenoid cavity
tuberosity ) near wrist ( ulnar notch , styloid process , Lister 's tubercle ) Ulna near elbow ( tuberosity , olecranon , coronoid process , radial notch ,
trochlear notch ) near wrist ( styloid process ) Hand Carpal bones scaphoid lunate triquetral pisiform trapezium trapezoid capitate hamate hamulus Metacarpal
bones 1st 2nd 3rd 4th 5th Phalanges proximal intermediate distal Retrieved from `` https://en.wikipedia.org/w/index.php?title=Glenoid_cavity&oldid=805938319 ''
Categories : Wikipedia articles incorporating text from the 20th edition of Gray 's Anatomy ( 1918 ) Shoulder Scapula Hidden categories : Articles containing
Greek - language text Medicine infobox template using GraySubject or GrayPage Medicine infobox template using Dorlands parameter Talk Contents About Wikipedia
Deutsch Español فارسی Français Galego Ido Italiano Magyar Português Svenska ไทย Українська Edit links This page was last edited on 18 October 2017 ,
</item 7>

<item 8>
Graph query row:
entity: glenoid cavity
entity_types: ['__KGBuilder__', '__Entity__', 'Thing']
entity_kind: None
relationship: LOCATED_IN
related_entity: lateral angle of the scapula
related_types: ['__KGBuilder__', '__Entity__', 'Thing']
related_kind: None
detail: None
</item 8>

<item 9>
Glenoid cavity
This reflects the importance of overhead limb postures and suggests a retention of arboreal adaptations in these hominoid primates , whereas the lateral
orientation of the glenoid in modern humans reflects the typical lowered position of the arm . In dinosaurs ( edit ) In dinosaurs the main bones of the pectoral
girdle were the scapula ( shoulder blade ) and the coracoid , both of which directly articulated with the clavicle . The place on the scapula where it
articulated with the humerus ( upper bone of the forelimb ) is called the glenoid . The glenoid is important because it defines the range of motion of the
humerus . Additional images ( edit ) Left scapula . Glenoid cavity shown in red . Animation . Glenoid cavity shown in red . Glenoid fossa of right side .
Anterior view of left scapula showing glenoid cavity `` 2 '' Animation of shoulder joint . The muscles shown are subscapularis muscle ( at right ) ,
infraspinatus muscle ( at top left ) , teres minor muscle ( at bottom left ) .
</item 9>

<item 10>
Glenoid cavity
Strong glenohumeral ligaments and muscles prevents dislocation in most cases . By being so shallow the glenoid cavity allows the shoulder joint to have the
greatest mobility of all joints in the body , allowing 120 degrees of unassisted flexion . Additional range of motion in shoulder flexion ( typically up to 180
degrees in humans ) is also accomplished by the great mobility of the scapula ( shoulder blade ) through a process known as scapulohumeral rhythm . Contents (
hide ) 1 Evolution 1.1 In dinosaurs 2 Additional images 3 See also 4 Notes 5 References 6 References 7 External links Evolution ( edit ) Interpretations of the
fossil remains of Australopithecus africanas ( STS 7 ) and A. afarensis ( AL 288 - 1 ; a.k.a. Lucy ) suggest that the glenoid fossa was oriented more cranially
in these species than in modern humans .
</item 10>
=== END ITEM id=bfa9172e1daaecc3 ===

=== ITEM id=f4ef1a920396947d ===
Gold sentences and facts:
Sentence 0: City Image Country Population Year Mexico City Mexico 8,918,653 2015 New York City United States 8,550,405 2015 Los Angeles United States 3,971,883 2015 Toronto Canada 2,826,498 2015 Chicago United States 2,720,546 2015 6 Houston United States 2,296,224 2015 7 Havana Cuba 2,117,625 2015 8 Montreal Canada 1,753,034 2015 9 Ecatepec de Morelos Mexico 1,677,678 2015 10 Philadelphia United States 1,567,442 2015 11 Phoenix United States 1,563,025 2015 12 San Antonio United States 1,469,845 2015 13 Guadalajara Mexico 1,460,148 2015 14 Puebla Mexico 1,437,939 2015 15 San Diego United States 1,394,928 2015 16 Juárez Mexico 1,382,753 2015 17 León Mexico 1,349,224 2015 18 Dallas United States 1,300,092 2015 19 Tijuana Mexico 1,298,475 2015 20 Calgary Canada 1,230,915 2015 21 Tegucigalpa Honduras 1,190,230 2015 22 Zapopan Mexico 1,179,681 2015 23 Monterrey Mexico 1,109,171 2015 24 Managua Nicaragua 1,048,134 2015 25 Nezahualcóyotl Mexico 1,039,867 2015 26 San Jose United States 1,026,908 2015 27 Santo Domingo Dominican Republic 1,007,997 2015 28 Guatemala City Guatemala 994,078 2015 29 Port - au - Prince Haiti 987,310 2015 30 Naucalpan Mexico 970,012 2015 31 Ottawa Canada 956,710 2015 32 Austin United States 931,830 2015 33 Edmonton Canada 899,447 2016 34 Mérida Mexico 892,363 2015 35 Querétaro Mexico 878,931 2015 36 Toluca Mexico 873,536 2015 37 Jacksonville United States 868,031 2015 38 Chihuahua Mexico 867,736 2015 39 San Francisco United States 864,816 2015 40 Indianapolis United States 853,173 2015 41 Columbus United States 850,106 2015 42 Fort Worth United States 833,319 2015 43 Charlotte United States 827,097 2015 44 Hermosillo Mexico 819,999 2015 45 Saltillo Mexico 791,074 2015 46 Aguascalientes Mexico 785,942 2015 47 Mississauga Canada 761,300 2015 48 San Luis Potosí Mexico 759,854 2015 49 Veracruz Mexico 752,171 2015 50 San Pedro Sula Honduras 742,118 2015 51 Santiago de los Caballeros Dominican Republic 723,162 2015 52 Culiacán Mexico 720,798 2015 53 Winnipeg Canada 718,400 2015 54 Mexicali Mexico 717,628 2015 55 Cancún Mexico 711,682 2015 56 Acapulco Mexico 710,261 2015 57 Tlalnepantla Mexico 700,734 2015 58 Seattle United States 684,451 2015 59 Guadalupe Mexico 682,880 2015 60 Denver United States 682,545 2015 61 El Paso United States 681,124 2015 62 Chimalhuacán Mexico 679,811 2015 63 Detroit United States 677,116 2015 64 Washington , D.C. United States 672,228 2015 65 Kingston Jamaica 669,627 2015 66 Boston United States 667,137 2015 67 Tlaquepaque Mexico 664,193 2015 68 Memphis United States 655,770 2015 69 Nashville United States 654,610 2015 70 Torreón Mexico 653,150 2015 71 Vancouver Canada 648,608 2015 72 Portland United States 632,309 2015 73 Reynosa Mexico 631,611 2015 74 Oklahoma City United States 631,346 2015 75 Las Vegas United States 623,747 2015 76 Baltimore United States 621,849 2015 77 Brampton Canada 620,700 2016 78 Louisville United States 615,366 2015 79 Morelia Mexico 608,190 2015 80 Milwaukee United States 600,155 2015 81 Tuxtla Gutiérrez Mexico 598,710 2015 82 Apodaca Mexico 597,207 2015 83 Durango Mexico 590,893 2015 84 Villa Nueva Guatemala 564,686 2015 85 Albuquerque United States 559,121 2015 86 Hamilton Canada 556,359 2015 87 Quebec City Canada 540,994 2015 88 Tonalá Mexico 536,111 2015 89 Tucson United States 531,641 2015 90 Cuautitlán Izcalli Mexico 531,041 2015 91 Surrey Canada 526,004 2015 92 Ciudad López Mateos Mexico 523,296 2015 93 Tultitlán Mexico 520,557 2015 94 Fresno United States 520,052 2015 95 Carrefour Haiti 501,768 2015
  Fact 0.0: Mexico City is a city in Mexico with a population of 8,918,653 in 2015.
  Fact 0.1: New York City is a city in the United States with a population of 8,550,405 in 2015.
  Fact 0.2: Los Angeles is a city in the United States with a population of 3,971,883 in 2015.
  Fact 0.3: Toronto is a city in Canada with a population of 2,826,498 in 2015.
  Fact 0.4: Chicago is a city in the United States with a population of 2,720,546 in 2015.
  Fact 0.5: Houston is a city in the United States, ranked 6th, with a population of 2,296,224 in 2015.
  Fact 0.6: Havana is a city in Cuba, ranked 7th, with a population of 2,117,625 in 2015.
  Fact 0.7: Montreal is a city in Canada, ranked 8th, with a population of 1,753,034 in 2015.
  Fact 0.8: Ecatepec de Morelos is a city in Mexico, ranked 9th, with a population of 1,677,678 in 2015.
  Fact 0.9: Philadelphia is a city in the United States, ranked 10th, with a population of 1,567,442 in 2015.
  Fact 0.10: Phoenix is a city in the United States, ranked 11th, with a population of 1,563,025 in 2015.
  Fact 0.11: San Antonio is a city in the United States, ranked 12th, with a population of 1,469,845 in 2015.
  Fact 0.12: Guadalajara is a city in Mexico, ranked 13th, with a population of 1,460,148 in 2015.
  Fact 0.13: Puebla is a city in Mexico, ranked 14th, with a population of 1,437,939 in 2015.
  Fact 0.14: San Diego is a city in the United States, ranked 15th, with a population of 1,394,928 in 2015.
  Fact 0.15: Juárez is a city in Mexico, ranked 16th, with a population of 1,382,753 in 2015.
  Fact 0.16: León is a city in Mexico, ranked 17th, with a population of 1,349,224 in 2015.
  Fact 0.17: Dallas is a city in the United States, ranked 18th, with a population of 1,300,092 in 2015.
  Fact 0.18: Tijuana is a city in Mexico, ranked 19th, with a population of 1,298,475 in 2015.
  Fact 0.19: Calgary is a city in Canada, ranked 20th, with a population of 1,230,915 in 2015.
  Fact 0.20: Tegucigalpa is a city in Honduras, ranked 21st, with a population of 1,190,230 in 2015.
  Fact 0.21: Zapopan is a city in Mexico, ranked 22nd, with a population of 1,179,681 in 2015.
  Fact 0.22: Monterrey is a city in Mexico, ranked 23rd, with a population of 1,109,171 in 2015.
  Fact 0.23: Managua is a city in Nicaragua, ranked 24th, with a population of 1,048,134 in 2015.
  Fact 0.24: Nezahualcóyotl is a city in Mexico, ranked 25th, with a population of 1,039,867 in 2015.
  Fact 0.25: San Jose is a city in the United States, ranked 26th, with a population of 1,026,908 in 2015.
  Fact 0.26: Santo Domingo is a city in the Dominican Republic, ranked 27th, with a population of 1,007,997 in 2015.
  Fact 0.27: Guatemala City is a city in Guatemala, ranked 28th, with a population of 994,078 in 2015.
  Fact 0.28: Port-au-Prince is a city in Haiti, ranked 29th, with a population of 987,310 in 2015.
  Fact 0.29: Naucalpan is a city in Mexico, ranked 30th, with a population of 970,012 in 2015.
  Fact 0.30: Ottawa is a city in Canada, ranked 31st, with a population of 956,710 in 2015.
  Fact 0.31: Austin is a city in the United States, ranked 32nd, with a population of 931,830 in 2015.
  Fact 0.32: Edmonton is a city in Canada, ranked 33rd, with a population of 899,447 in 2016.
  Fact 0.33: Mérida is a city in Mexico, ranked 34th, with a population of 892,363 in 2015.
  Fact 0.34: Querétaro is a city in Mexico, ranked 35th, with a population of 878,931 in 2015.
  Fact 0.35: Toluca is a city in Mexico, ranked 36th, with a population of 873,536 in 2015.
  Fact 0.36: Jacksonville is a city in the United States, ranked 37th, with a population of 868,031 in 2015.
  Fact 0.37: Chihuahua is a city in Mexico, ranked 38th, with a population of 867,736 in 2015.
  Fact 0.38: San Francisco is a city in the United States, ranked 39th, with a population of 864,816 in 2015.
  Fact 0.39: Indianapolis is a city in the United States, ranked 40th, with a population of 853,173 in 2015.
  Fact 0.40: Columbus is a city in the United States, ranked 41st, with a population of 850,106 in 2015.
  Fact 0.41: Fort Worth is a city in the United States, ranked 42nd, with a population of 833,319 in 2015.
  Fact 0.42: Charlotte is a city in the United States, ranked 43rd, with a population of 827,097 in 2015.
  Fact 0.43: Hermosillo is a city in Mexico, ranked 44th, with a population of 819,999 in 2015.
  Fact 0.44: Saltillo is a city in Mexico, ranked 45th, with a population of 791,074 in 2015.
  Fact 0.45: Aguascalientes is a city in Mexico, ranked 46th, with a population of 785,942 in 2015.
  Fact 0.46: Mississauga is a city in Canada, ranked 47th, with a population of 761,300 in 2015.
  Fact 0.47: San Luis Potosí is a city in Mexico, ranked 48th, with a population of 759,854 in 2015.
  Fact 0.48: Veracruz is a city in Mexico, ranked 49th, with a population of 752,171 in 2015.
  Fact 0.49: San Pedro Sula is a city in Honduras, ranked 50th, with a population of 742,118 in 2015.
  Fact 0.50: Santiago de los Caballeros is a city in the Dominican Republic, ranked 51st, with a population of 723,162 in 2015.
  Fact 0.51: Culiacán is a city in Mexico, ranked 52nd, with a population of 720,798 in 2015.
  Fact 0.52: Winnipeg is a city in Canada, ranked 53rd, with a population of 718,400 in 2015.
  Fact 0.53: Mexicali is a city in Mexico, ranked 54th, with a population of 717,628 in 2015.
  Fact 0.54: Cancún is a city in Mexico, ranked 55th, with a population of 711,682 in 2015.
  Fact 0.55: Acapulco is a city in Mexico, ranked 56th, with a population of 710,261 in 2015.
  Fact 0.56: Tlalnepantla is a city in Mexico, ranked 57th, with a population of 700,734 in 2015.
  Fact 0.57: Seattle is a city in the United States, ranked 58th, with a population of 684,451 in 2015.
  Fact 0.58: Guadalupe is a city in Mexico, ranked 59th, with a population of 682,880 in 2015.
  Fact 0.59: Denver is a city in the United States, ranked 60th, with a population of 682,545 in 2015.
  Fact 0.60: El Paso is a city in the United States, ranked 61st, with a population of 681,124 in 2015.
  Fact 0.61: Chimalhuacán is a city in Mexico, ranked 62nd, with a population of 679,811 in 2015.
  Fact 0.62: Detroit is a city in the United States, ranked 63rd, with a population of 677,116 in 2015.
  Fact 0.63: Washington, D.C. is a city in the United States, ranked 64th, with a population of 672,228 in 2015.
  Fact 0.64: Kingston is a city in Jamaica, ranked 65th, with a population of 669,627 in 2015.
  Fact 0.65: Boston is a city in the United States, ranked 66th, with a population of 667,137 in 2015.
  Fact 0.66: Tlaquepaque is a city in Mexico, ranked 67th, with a population of 664,193 in 2015.
  Fact 0.67: Memphis is a city in the United States, ranked 68th, with a population of 655,770 in 2015.
  Fact 0.68: Nashville is a city in the United States, ranked 69th, with a population of 654,610 in 2015.
  Fact 0.69: Torreón is a city in Mexico, ranked 70th, with a population of 653,150 in 2015.
  Fact 0.70: Vancouver is a city in Canada, ranked 71st, with a population of 648,608 in 2015.
  Fact 0.71: Portland is a city in the United States, ranked 72nd, with a population of 632,309 in 2015.
  Fact 0.72: Reynosa is a city in Mexico, ranked 73rd, with a population of 631,611 in 2015.
  Fact 0.73: Oklahoma City is a city in the United States, ranked 74th, with a population of 631,346 in 2015.
  Fact 0.74: Las Vegas is a city in the United States, ranked 75th, with a population of 623,747 in 2015.
  Fact 0.75: Baltimore is a city in the United States, ranked 76th, with a population of 621,849 in 2015.
  Fact 0.76: Brampton is a city in Canada, ranked 77th, with a population of 620,700 in 2016.
  Fact 0.77: Louisville is a city in the United States, ranked 78th, with a population of 615,366 in 2015.
  Fact 0.78: Morelia is a city in Mexico, ranked 79th, with a population of 608,190 in 2015.
  Fact 0.79: Milwaukee is a city in the United States, ranked 80th, with a population of 600,155 in 2015.
  Fact 0.80: Tuxtla Gutiérrez is a city in Mexico, ranked 81st, with a population of 598,710 in 2015.
  Fact 0.81: Apodaca is a city in Mexico, ranked 82nd, with a population of 597,207 in 2015.
  Fact 0.82: Durango is a city in Mexico, ranked 83rd, with a population of 590,893 in 2015.
  Fact 0.83: Villa Nueva is a city in Guatemala, ranked 84th, with a population of 564,686 in 2015.
  Fact 0.84: Albuquerque is a city in the United States, ranked 85th, with a population of 559,121 in 2015.
  Fact 0.85: Hamilton is a city in Canada, ranked 86th, with a population of 556,359 in 2015.
  Fact 0.86: Quebec City is a city in Canada, ranked 87th, with a population of 540,994 in 2015.
  Fact 0.87: Tonalá is a city in Mexico, ranked 88th, with a population of 536,111 in 2015.
  Fact 0.88: Tucson is a city in the United States, ranked 89th, with a population of 531,641 in 2015.
  Fact 0.89: Cuautitlán Izcalli is a city in Mexico, ranked 90th, with a population of 531,041 in 2015.
  Fact 0.90: Surrey is a city in Canada, ranked 91st, with a population of 526,004 in 2015.
  Fact 0.91: Ciudad López Mateos is a city in Mexico, ranked 92nd, with a population of 523,296 in 2015.
  Fact 0.92: Tultitlán is a city in Mexico, ranked 93rd, with a population of 520,557 in 2015.
  Fact 0.93: Fresno is a city in the United States, ranked 94th, with a population of 520,052 in 2015.
  Fact 0.94: Carrefour is a city in Haiti, ranked 95th, with a population of 501,768 in 2015.

Retrieved items:
<item 1>
List of North American cities by population
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 1>

<item 2>
List of North American cities by population
The April 2016 census result is given instead . Contents ( hide ) 1 List 2 See also 3 Notes 4 References List ( edit ) City Image Country Population Year Mexico
City Mexico 8,918,653 2015 New York City United States 8,550,405 2015 Los Angeles United States 3,971,883 2015 Toronto Canada 2,826,498 2015 Chicago United
States 2,720,546 2015 6 Houston United States 2,296,224 2015 7 Havana Cuba 2,117,625 2015 8 Montreal Canada 1,753,034 2015 9 Ecatepec de Morelos Mexico
1,677,678 2015 10 Philadelphia United States 1,567,442 2015 11 Phoenix United States 1,563,025 2015 12 San Antonio United States 1,469,845 2015 13 Guadalajara
Mexico 1,460,148 2015 14 Puebla Mexico 1,437,939 2015 15 San Diego United States 1,394,
</item 2>

<item 3>
List of North American cities by population
INEGI . p. 5 . Retrieved 2016 - 11 - 08 . ^ Jump up to : `` 2015 Municipal Affairs Population List '' ( PDF ) . Alberta Municipal Affairs . ISBN 978 - 1 - 4601
- 2630 - 1 . Retrieved July 17 , 2016 . ^ Jump up to : `` Encuesta Intercensal 2015 : Tabulados - descarga '' . INEGI . Archived from the original on 2016 - 11
- 13 . Retrieved 2016 - 11 - 08 . ^ Jump up to : Brinkhoff , Thomas . `` Mexico : Administrative Division '' . City Population . Retrieved 2016 - 11 - 08 . ^
Jump up to : `` Annual Estimates of the Resident Population for Incorporated Places of 50,000 or More , Ranked by July 1 , 2015 Population : April 1 , 2010 to
July 1 , 2015 '' . United States Census Bureau , Population Division . May 2016 .
</item 3>

<item 4>
List of North American cities by population
310 2015 30 Naucalpan Mexico 970,012 2015 31 Ottawa Canada 956,710 2015 32 Austin United States 931,830 2015 33 Edmonton Canada 899,447 2016 34 Mérida Mexico
892,363 2015 35 Querétaro Mexico 878,931 2015 36 Toluca Mexico 873,536 2015 37 Jacksonville United States 868,031 2015 38 Chihuahua Mexico 867,736 2015 39 San
Francisco United States 864,816 2015 40 Indianapolis United States 853,173 2015 41 Columbus United States 850,106 2015 42 Fort Worth United States 833,319 2015
43 Charlotte United States 827,097 2015 44 Hermosillo Mexico 819,999 2015 45 Saltillo Mexico 791,074 2015 46 Aguascalientes Mexico 785,942 2015 47 Mississauga
Canada 761,300 2015 48 San Luis Potosí Mexico 759,
</item 4>

<item 5>
List of North American cities by population
United States Census Bureau , Population Division . May 2016 . Retrieved 2016 - 05 - 19 . Jump up ^ `` Standard Geographical Classification ( SGC ) 2006 :
Economic Regions ( 3520 - Toronto , census subdivisions ) '' . Statistics Canada . 2014 - 04 - 01 . Archived from the original on 2015 - 05 - 30 . Retrieved
2015 - 06 - 11 . ^ Jump up to : `` Table 3.5 : Population and demographic factors of growth by census division , provinces and territories '' . Statistics
Canada . 2016 - 02 - 10 . Retrieved 2016 - 04 - 24 . Jump up ^ `` 3 . Población Proyectada al 30 de junio , por edades simples y años simples . AMBOS SEXOS .
Período 2015 - 2050 '' ( PDF ) . Proyecciones de la Población de Cuba 2015 - 2050 . Oficina Nacional de Estadísticas .
</item 5>

<item 6>
List of North American cities by population
C. United States 672,228 2015 65 Kingston Jamaica 669,627 2015 66 Boston United States 667,137 2015 67 Tlaquepaque Mexico 664,193 2015 68 Memphis United States
655,770 2015 69 Nashville United States 654,610 2015 70 Torreón Mexico 653,150 2015 71 Vancouver Canada 648,608 2015 72 Portland United States 632,309 2015 73
Reynosa Mexico 631,611 2015 74 Oklahoma City United States 631,346 2015 75 Las Vegas United States 623,747 2015 76 Baltimore United States 621,849 2015 77
Brampton Canada 620,700 2016 78 Louisville United States 615,366 2015 79 Morelia Mexico 608,190 2015 80 Milwaukee United States 600,155 2015 81 Tuxtla Gutiérrez
Mexico 598,710 2015 82 Apodaca Mexico 597,207 2015 83 Durango Mexico 590,
</item 6>

<item 7>
List of North American cities by population
437,939 2015 15 San Diego United States 1,394,928 2015 16 Juárez Mexico 1,382,753 2015 17 León Mexico 1,349,224 2015 18 Dallas United States 1,300,092 2015 19
Tijuana Mexico 1,298,475 2015 20 Calgary Canada 1,230,915 2015 21 Tegucigalpa Honduras 1,190,230 2015 22 Zapopan Mexico 1,179,681 2015 23 Monterrey Mexico
1,109,171 2015 24 Managua Nicaragua 1,048,134 2015 25 Nezahualcóyotl Mexico 1,039,867 2015 26 San Jose United States 1,026,908 2015 27 Santo Domingo Dominican
Republic 1,007,997 2015 28 Guatemala City Guatemala 994,078 2015 29 Port - au - Prince Haiti 987,310 2015 30 Naucalpan Mexico 970,
</item 7>

<item 8>
List of North American cities by population
List of North American cities by Population - wikipedia List of North American cities by Population Jump to : navigation , search For the majority of cities in
North America , the most recent official population census results , estimates or short - term projections date to 2015 . This list compiles figures for all
North American cities with a population within city limits exceeding 500,000 that year . These figures do not reflect the population of the urban agglomeration
or metropolitan area which typically do not coincide with the administrative boundaries of the city . They refer to mid-2015 populations with the following
exceptions : Mexican cities , whose figures derive from the 2015 Intercensal Survey conducted by INEGI with a reference date of 15 March 2015 ; Calgary , whose
2015 municipal census had a reference date of April 1 ; Edmonton , which conducted municipal censuses in April 2014 and April 2016 but has not reported an
intercensal estimate of its 2015 population . The April 2016 census result is given instead .
</item 8>

<item 9>
List of North American cities by population
300 2015 48 San Luis Potosí Mexico 759,854 2015 49 Veracruz Mexico 752,171 2015 50 San Pedro Sula Honduras 742,118 2015 51 Santiago de los Caballeros Dominican
Republic 723,162 2015 52 Culiacán Mexico 720,798 2015 53 Winnipeg Canada 718,400 2015 54 Mexicali Mexico 717,628 2015 55 Cancún Mexico 711,682 2015 56 Acapulco
Mexico 710,261 2015 57 Tlalnepantla Mexico 700,734 2015 58 Seattle United States 684,451 2015 59 Guadalupe Mexico 682,880 2015 60 Denver United States 682,545
2015 61 El Paso United States 681,124 2015 62 Chimalhuacán Mexico 679,811 2015 63 Detroit United States 677,116 2015 64 Washington , D.C. United States 672,228
2015 65 Kingston Jamaica 669,
</item 9>

<item 10>
List of North American cities by population
Retrieved 2016 - 05 - 01 . ( hide ) World 's largest cities City proper World ( Capitals ) Americas ( North Latin Central South ) Asia ( Arab world ASEAN Arabia
Far East ) Europe ( European Union Nordic ) Metropolitan area World Americas ( North South West Indies ) Europe ( European Union Nordic ) Asia Africa
Transborder Aggregate urban area † World Europe ( European Union Nordic ) Asia Urban agglomeration † North America Europe ( European Union ) Asia Historical
World Europe Related articles Global city Megacity Megalopolis Conurbation Suburbs † Aggregate urban areas are entire continuous built - up areas , grouping
together contiguous cities ( known as conurbations ) , whereas urban agglomerations are the built - up areas of individual cities . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=List_of_North_American_cities_by_population&oldid=822737139 '' Categories : Lists of cities by population Lists of
cities in the Americas North America - related lists Cities in North America Talk Contents About Wikipedia Español اردو 中文 Edit links This page was last edited
on 28 January 2018 , at 05 : 43 .
</item 10>
=== END ITEM id=f4ef1a920396947d ===

=== ITEM id=221064d8b61ea46b ===
Gold sentences and facts:
Sentence 0: In 1918 , Firestone Tire and Rubber Company of Canada was incorporated in Hamilton , Ontario and in 1922 , the first Canadian - made tire rolled off the line on September 15 . During the 1920s , Firestone produced the Oldfield tire , named for racing driver Barney Oldfield .
  Fact 0.0: Firestone Tire and Rubber Company of Canada was incorporated in 1918.
  Fact 0.1: Firestone Tire and Rubber Company of Canada was incorporated in Hamilton, Ontario.
  Fact 0.2: In 1922, the first Canadian-made tire rolled off the line on September 15.
  Fact 0.3: During the 1920s, Firestone produced the Oldfield tire.
  Fact 0.4: The Oldfield tire was named for racing driver Barney Oldfield.
Sentence 1: Bridgestone / Firestone also produces tires under the Fuzion brand name . Fuzion is a tire manufactured in China for Bridgestone / Firestone as a budget replacement tire .
  Fact 1.0: Bridgestone / Firestone also produces tires under the Fuzion brand name.
  Fact 1.1: Fuzion is a tire manufactured in China for Bridgestone / Firestone.
  Fact 1.2: Fuzion is a budget replacement tire.

Retrieved items:
<item 1>
Firestone Tire and Rubber Company
Friends say Ford wanted to be near the ocean and picked Long Beach and suggested Firestone go to South Gate , California . The tiny community southeast of
Downtown was mostly agriculture at the time and Firestone found 40 acres of beanfield to house his new manufacturing plant . Architects Curlett and Beelman
created a spectacular four - storey Italianate complex , with its own power plant and gorgeous polychrome murals by Gladding McBean depicting the tire and
rubber - making process . A year after the plant opened in 1928 it doubled in size . By 1954 , when they added the Corporal guided missile to their offerings ,
the plant was nearly a million square feet . The town grew around Firestone , they named the main boulevard through town after Harvey , and Los Angeles became
the number one tire market in the country . By the mid-1970s Ford and GM had massive layoffs as Firestone and other manufacturers opened new plants in non-union
locales like Wilson , North Carolina . After much downsizing the end came in 1980 when 1,300 workers were laid off and the plant closed .
</item 1>

<item 2>
Firestone Tire and Rubber Company
Harvey Firestone had a friendship with Henry Ford . Firestone used this relationship to become the original equipment supplier of Ford Motor Company automobiles
, and was also active in the replacement market . In 1988 , the company was sold to the Japanese Bridgestone Corporation . Contents ( hide ) 1 History 1.1 Early
- to - mid-20th century 1.2 Restructuring and sale to Bridgestone 2 In motorsport 3 Corporate troubles 3.1 Great American streetcar scandal 3.2 Firestone 500
tread separation problem 3.3 Liberian rubber plantation 3.4 Ford Explorer rollover problem 4 Television advertisement jingle 5 Other brands 6 See also 7
References 8 External links History ( edit ) Early - to - mid-20th century ( edit ) The First Firestone store Firestone newspaper ad from 1919 , explaining the
benefits of their cord tire . Firestone was originally based in Akron , Ohio , also the hometown of its archrival , Goodyear Tire and Rubber Company , and
another two mid-sized competitors , General Tire and Rubber and BF Goodrich .
</item 2>

<item 3>
Firestone Tire and Rubber Company
East Los Angeles College has proposed a new satellite campus at the site . In 1928 the company built a factory in Brentford , England , a longtime Art Deco
landmark on a major route into the city . This closed in 1979 . In 1936 the company opened a plant in Memphis , Tennessee . With a work force exceeding 3,000
employees , the Memphis plant was the largest tire manufacturer in the company 's worldwide operation . On July 1 , 1963 , the company celebrated the production
of 100 million tires in Memphis . The plant was closed in 1982 . During World War II the company was called on by the U.S. Government to make artillery shells ,
aluminum kegs for food transport and rubberized military products . Firestone ranked 55th among United States corporations in the value of World War II military
production contracts . In the 1940s , Firestone was given a defense contract to produce plastic helmet liners . While outproduced by Westinghouse Electric they
still made a fair amount for the M1 Helmet .
</item 3>

<item 4>
Firestone Tire and Rubber Company
The combined Bridgestone / Firestone North American operations are now based in Nashville , Tennessee . In 2008 , the companies celebrated a 20 - year
anniversary of the merger , and changed the tire division name to Bridgestone Americas Tire Operations , LLC . In 2012 , Bridgestone Americas opened a $100
million technical center in Akron . Firestone tires used for the Indianapolis 500 race . Apart from tires , several companies and divisions operate with the
Firestone brand in its name . These companies include Firestone Building Products , LLC , Firestone Industrial Products , LLC , Firestone Complete Auto Care ,
Firestone Natural Rubber Company and Firestone Specialty Products . Firestone Building Products and Firestone Industrial Products are headquartered in
Indianapolis , Indiana and operate in 23 U.S. states and ten countries , with 11,000 employees worldwide . The company reports annual sales of more than $2.5
billion . In 1972 , Firestone received a ten - year import `` concession '' by the Kenyan government to secure Firestone 's investment in a domestic tire plant
, which gave it a virtual monopoly . This included both general price and foreign exchange controls .
</item 4>

<item 5>
Firestone Tire and Rubber Company
Founded on August 3 , 1900 , the company initiated operations with 12 employees . Together , Firestone and Goodyear were the largest suppliers of automotive
tires in North America for over 75 years . In 1906 Henry Ford chose Firestone for Model T original equipment tires . In 1918 , Firestone Tire and Rubber Company
of Canada was incorporated in Hamilton , Ontario and in 1922 , the first Canadian - made tire rolled off the line on September 15 . During the 1920s , Firestone
produced the Oldfield tire , named for racing driver Barney Oldfield . In 1926 , the company opened one of the world 's biggest rubber plantations in Liberia ,
West Africa , spanning more than 1 million acres . 1926 was also the year that the company opened its first Firestone Complete Auto Care store ( Firestone
Complete Auto Care is a division of Firestone that offers automotive maintenance and repair ) . In 1927 , Henry Ford and tire maven Harvey Firestone took a trip
to Los Angeles to select locations for their new factories .
</item 5>

<item 6>
Firestone Tire and Rubber Company
Firestone tire and Rubber company - wikipedia Firestone tire and Rubber company Jump to : navigation , search Firestone Tire and Rubber Company Type Subsidiary
Industry Manufacturing Founded Akron , Ohio , United States ( August 3 , 1900 ; 117 years ago ( 1900 - 08 - 03 ) ) Founder Harvey Firestone Headquarters 200 4th
Ave S , Nashville , Tennessee , United States Area served Worldwide Key people Gordon Knapp ( CEO and President ) Bill Thompson ( COO ) Products Tires Revenue
US $ 2.09 billion ( 2004 ) Number of employees 40,000 Parent Bridgestone ( 1988 -- present ) Website firestone.com The Firestone Tire and Rubber Company is an
American tire company founded by Harvey Samuel Firestone in 1900 to supply pneumatic tires for wagons , buggies , and other forms of wheeled transportation
common in the era . Firestone soon saw the huge potential for marketing tires for automobiles . The company was a pioneer in the mass production of tires .
Harvey Firestone had a friendship with Henry Ford .
</item 6>

<item 7>
Firestone Tire and Rubber Company
This included both general price and foreign exchange controls . When the 10 - year period came to an end in 1979 , Firestone retaliated by increasing
production , making entry less attractive . Headquarters eventually canceled expansion and failed negotiations lead to no further investments . In motorsport (
edit ) The 1911 Indianapolis 500 auto race was won by a car running Firestone tires . Firestone - shod cars won all editions of the race from 1920 to 1966 . The
company also provided tires to Formula One from 1950 to 1974 . Citing rising costs as the cause , Firestone announced its withdrawal from professional
motorsport in August 1974 with the exception of Formula 5000 and Formula One . The change was forecast to reduce Firestone 's annual race testing budget from
between $6 and $8 million dollars to around $500,000 . Shortly afterwards , Firestone extended this to a complete withdrawal by the end of the 1974 season . The
manufacturer returned in 1995 to the CART series with technical assistance from Bridgestone .
</item 7>

<item 8>
Firestone Tire and Rubber Company
External links ( edit ) Wikimedia Commons has media related to Firestone Tire and Rubber Company . Official website Century of Progress from the Summit Memory
Project Bridgestone Corporation Subsidiaries Firestone Diversified Products Firestone Tire and Rubber Company Firestone - Apsley Rubber Company Products
Bridgestone Picnica Helmz Facilities Firestone Tyre Factory People Harvey Samuel Firestone Related Akron Firestone Non-Skids Bridgestone Arena Bridgestone
Museum of Art Bridgestone Open Firestone and Ford tire controversy Firestone Country Club International Labor Rights Forum v Firestone Tire and Rubber Co The
Voice of Firestone WGC - Bridgestone Invitational Automotive industry in the United States Automotive industry Economy of the United States Transportation in
the United States American vehicle manufacturers ( list ) AGCO Challenger Tractor Massey Ferguson AM General American Expedition Vehicles American Growler Amp
Electric Vehicles Anteros Coachworks Arcimoto Armour Group , Inc .
</item 8>

<item 9>
Firestone Tire and Rubber Company
ISBN 978 - 0 - 15 - 692620 - 1 . Jump up ^ `` A Global Dream '' . Bridgestone Americas . Retrieved 3 August 2013 . Jump up ^ `` History of Bridgestone /
Firestone '' . Bridgestone / Firestone Canada . 2009 . Archived from the original on 10 May 2010 . Jump up ^ Firestone 8000 Tire Order by Ford Ford Chooses
Firestone for Model T Jump up ^ Canada manufactures first Firestone tire Archived 2010 - 05 - 10 at the Wayback Machine . Firestone Canada Incorporated Jump up
^ Chris Nichols ( 2013 - 03 - 01 ) . `` DispL. A. Case # 52 : L.A. 's First Firestone Tire - Los Angeles Magazine '' . Lamag.com . Retrieved 2015 - 12 - 17 .
Jump up ^ Lost Memphis Firestone Tire & Rubber Company Jump up ^ Firestone Closings Jump up ^ Peck , Merton J.
</item 9>

<item 10>
Firestone Tire and Rubber Company
Retrieved 2007 - 03 - 21 . Jump up ^ Isidore , Chris ( May 21 , 2001 ) . `` Firestone severs ties to Ford '' . CNN . Jump up ^ NHTSA Recall Campaign 01T - 016
Q4 Vehicle Safety Recall Quarterly Report Information , January 25 , 2008 ( Amended March 26 , 2009 ) ( from the National Highway Traffic Safety Administration
) ^ Jump up to : Channel 2 Investigation Finds Recalled Tires for Sale , November 4 , 2013 Archived November 5 , 2013 , at the Wayback Machine . Jump up ^ ``
Firestone Jingle '' ( QuickTime ) . Firestone . Retrieved 2007 - 03 - 22 . External links ( edit ) Wikimedia Commons has media related to Firestone Tire and
Rubber Company .
</item 10>
=== END ITEM id=221064d8b61ea46b ===
