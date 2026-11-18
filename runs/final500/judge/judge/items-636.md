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

This file has 11 items.

=== ITEM id=5b211ffa2c9ed790 ===
Gold sentences and facts:
Sentence 0: Electronic Attack Squadron 138 (VAQ-138), also known as the "Yellow Jackets", is an expeditionary EA-18G Growler squadron of the United States Navy based at Naval Air Station Whidbey Island, Washington.
  Fact 0.0: Electronic Attack Squadron 138 (VAQ-138) is also known as the "Yellow Jackets".
  Fact 0.1: Electronic Attack Squadron 138 is an expeditionary EA-18G Growler squadron of the United States Navy.
  Fact 0.2: Electronic Attack Squadron 138 is based at Naval Air Station Whidbey Island, Washington.
Sentence 1: The Boeing EA-18G Growler is an American carrier-based electronic warfare aircraft, a specialized version of the two-seat F/A-18F Super Hornet.
  Fact 1.0: The Boeing EA-18G Growler is an American carrier-based electronic warfare aircraft.
  Fact 1.1: The Boeing EA-18G Growler is a specialized version of the two-seat F/A-18F Super Hornet.

Retrieved items:
<item 1>
VFA-2
Strike Fighter Squadron 2 (VFA-2) also known as the "Bounty Hunters" is a United States Navy F/A-18F Super Hornet strike fighter squadron based at Naval Air
Station Lemoore, California.

 Their tail code is NE and their callsign is "Bullet".

 They are attached to Carrier Air Wing 2 (CVW-2), a composite unit made up of a wide array of aircraft performing a variety of combat and support missions that
deploy aboard the "Carl Vinson" .
</item 1>

<item 2>
VAQ-138
Electronic Attack Squadron 138 (VAQ-138), also known as the "Yellow Jackets", is an expeditionary EA-18G Growler squadron of the United States Navy based at
Naval Air Station Whidbey Island, Washington.

 Their radio callsign is "RAMPAGE".
</item 2>

<item 3>
Boeing F/A-18E/F Super Hornet
The Boeing F/A-18E and F/A-18F Super Hornet are twin-engine carrier-capable multirole fighter aircraft variants based on the McDonnell Douglas F/A-18 Hornet.

 The F/A-18E single-seat and F/A-18F tandem-seat variants are larger and more advanced derivatives of the F/A-18C and D Hornet.

 The Super Hornet has an internal 20 mm M61 rotary cannon and can carry air-to-air missiles and air-to-surface weapons.

 Additional fuel can be carried in up to five external fuel tanks and the aircraft can be configured as an airborne tanker by adding an external air refueling
system.
</item 3>

<item 4>
Type 002 aircraft carrier
The Type 002 aircraft carrier is a second generation Chinese aircraft carrier design.

 The ship will be the first Chinese aircraft carrier to be equipped with a catapult.

 According to the latest report, the construction of this carrier has been rescheduled for a decision on the catapults, where the latest competition results
suggested that steam catapults will be used in the Type 002 aircraft carrier.

 The addition of aircraft catapult will give China the ability to launch various fixed wing support aircraft like airborne early warning (AEW) and carrier
onboard delivery aircraft.

 Fighters can also be launched at full combat load and reach its full potential unlike current Type 001 aircraft carriers operated by Chinese navy.
</item 4>

<item 5>
No. 82 Wing RAAF
No. 82 Wing is the strike and reconnaissance wing of the Royal Australian Air Force (RAAF).

 It is headquartered at RAAF Base Amberley, Queensland.

 Coming under the control of Air Combat Group, the wing operates F/A-18F Super Hornet multirole fighters and Pilatus PC-9 forward air control aircraft.

 Its units include Nos. 1 and 6 Squadrons, operating the Super Hornet, and No. 4 Squadron, operating the PC-9.
</item 5>

<item 6>
VAQ-209
Electronic Attack Squadron 209 (VAQ-209) is a United States Navy Reserve electronic attack squadron.

 Known as the "Star Warriors", the squadron flies the EA-18G Growler carrier-based electronic warfare jet aircraft.

 Based at Naval Air Station Whidbey Island, WA, it is assigned to the Tactical Support Wing.
</item 6>

<item 7>
No. 1 Squadron RAAF
No. 1 Squadron is a Royal Australian Air Force (RAAF) squadron headquartered at RAAF Base Amberley, Queensland.

 Controlled by No. 82 Wing, it is equipped with Boeing F/A-18F Super Hornet multi-role fighters.

 The squadron was formed under the Australian Flying Corps in 1916 and saw action in the Sinai and Palestine Campaigns during World War I.

 It flew obsolete Royal Aircraft Factory B.E.2s, B.E.12s, Martinsyde G.100s and G.102s, as well as Airco DH.6s, Bristol Scouts and Nieuport 17s, before
re-equipping with the R.E.8 in October 1917 and finally the Bristol Fighter in December.

 Its commanding officer in 1917–18 was Major Richard Williams, later known as the "Father of the RAAF".

 Disbanded in 1919, No. 1 Squadron was re-formed on paper as part of the RAAF in 1922, and re-established as an operational unit three years later.
</item 7>

<item 8>
Boeing EA-18G Growler
The Boeing EA-18G Growler is an American carrier-based electronic warfare aircraft, a specialized version of the two-seat F/A-18F Super Hornet.

 The EA-18G replaced the Northrop Grumman EA-6B Prowlers in service with the United States Navy.

 The Growler's electronic warfare capability is primarily provided by Northrop Grumman.

 The EA-18G began production in 2007 and entered operational service with the US Navy in late 2009.

 Australia has also purchased twelve EA-18Gs, which entered service with the Royal Australian Air Force in 2017.
</item 8>

<item 9>
VAQ-135
Electronic Attack Squadron 135 (VAQ-135), known as the "Black Ravens", is a United States Navy electronic attack squadron that currently operates the EA-18G
Growler carrier-based electronic warfare jet aircraft.

 The squadron is permanently stationed at Naval Air Station Whidbey Island with a radio callsign of ""Thunder"".
</item 9>

<item 10>
VFA-154
Strike Fighter Squadron 154 (VFA-154), also known as the "Black Knights", is a United States Navy strike fighter squadron stationed at Naval Air Station
Lemoore.

 The Black Knights are an operational fleet squadron flying the F/A-18F Super Hornet.

 They are currently attached to Carrier Air Wing Eleven and deployed aboard the aircraft carrier USS "Nimitz" .

 Their tailcode is NH and their callsign is "Knight".
</item 10>
=== END ITEM id=5b211ffa2c9ed790 ===

=== ITEM id=9fbdc9575af8f321 ===
Gold sentences and facts:
Sentence 0:  Reed, along with Mike Pringle and Johnny Bright, is one of the players most often mentioned as being the greatest running back in CFL history.
  Fact 0.0: Reed is one of the players most often mentioned as being the greatest running back in CFL history.
  Fact 0.1: Mike Pringle is one of the players most often mentioned as being the greatest running back in CFL history.
  Fact 0.2: Johnny Bright is one of the players most often mentioned as being the greatest running back in CFL history.
Sentence 1: Johnny D. Bright (June 11, 1930 – December 14, 1983) was a professional Canadian football player in the Canadian Football League.
  Fact 1.0: Johnny D. Bright was born June 11, 1930.
  Fact 1.1: Johnny D. Bright died December 14, 1983.
  Fact 1.2: Johnny D. Bright was a professional Canadian football player.
  Fact 1.3: Johnny D. Bright played in the Canadian Football League.
Sentence 2:  He played college football at Drake University.
  Fact 2.0: Johnny D. Bright played college football at Drake University.

Retrieved items:
<item 1>
Johnny Bright
Johnny D. Bright (June 11, 1930 – December 14, 1983) was a professional Canadian football player in the Canadian Football League.

 He played college football at Drake University.

 He is a member of the Canadian Football Hall of Fame, the National Football Foundation's College Football Hall of Fame, the Missouri Valley Conference Hall of
Fame, the Edmonton Eskimos Wall of Honour, the Alberta Sports Hall of Fame, and the "Des Moines Register's" Iowa Sports Hall of Fame.
</item 1>

<item 2>
Dan Regan
Dan Regan is a former trombone player for the Southern California-based ska punk band Reel Big Fish.

 He joined the band in 1994 and retired in October 2013.

 Regan has also played trombone in The Littlest Man Band with Scott Klopfenstein (formerly of Reel Big Fish) and currently has a hip-hop side project under the
alias Black Casper.

 He was the second longest lasting member of Reel Big Fish from 1994 until October 2013.

 He currently is starting a brewery in Long Beach, California.
</item 2>

<item 3>
Wilbert Montgomery
Wilbert Montgomery (born September 16, 1954) is a former American football player in the National Football League for nine years with the Philadelphia Eagles
and Detroit Lions.

 In the past, Montgomery has been the running backs and tight ends coach for St. Louis Rams (1997–2005), the running backs coach for the Detroit Lions
(2006-2007), the running backs coach for the Baltimore Ravens (2008–2013), and the running backs coach for the Cleveland Browns (2014–2015).
</item 3>

<item 4>
Skip Peete
Skip Peete (born January 30, 1963) is an American football coach who serves as the current running backs coach for the Los Angeles Rams of the National Football
League (NFL).

 Prior to the Rams, he coached running backs for the Chicago Bears, Dallas Cowboys, and Oakland Raiders.

 He also has experience in college, coaching running backs for two years at UCLA, wide receivers at Michigan State and Rutgers, and both running backs and wide
receivers at the University of Pittsburgh.
</item 4>

<item 5>
George Reed (Canadian football)
George Robert Reed, CM, SOM (born October 2, 1939), is a former American college football and Canadian Football League player.

 Reed, along with Mike Pringle and Johnny Bright, is one of the players most often mentioned as being the greatest running back in CFL history.

 In November 2006, Reed was voted one of the CFL's Top 50 players (#2) of the league's modern era by Canadian sports network TSN.
</item 5>

<item 6>
Damon Allen
Damon Allen (born July 29, 1963) is a former professional quarterback who played in the Canadian Football League.

 He is currently second in all-time professional football passing yards and second in all-time CFL passing yards after he was surpassed for first place by the
Montréal Alouettes' Anthony Calvillo on October 10, 2011.

 He is pro-football's combined passing & rushing leader with a total of 84,301 yards.

 Allen retired as professional football's all-time leading passer with 72,381 passing yards after he surpassed Warren Moon's total of 70,553 yards (in both the
CFL and NFL combined) on September 4, 2006 in the annual Labour Day Classic.

 He also retired in third place in all-time CFL rushing yards with 11,920 yards, behind Mike Pringle and George Reed.

 The 2007 season marked Allen's twenty-third season in the CFL and he officially announced his retirement on May 28, 2008 at age 44.

 Allen is the younger brother of Pro Football Hall of Famer Marcus Allen.
</item 6>

<item 7>
2,000-yard club
The 2,000-yard club is a group of seven National Football League (NFL) running backs that have rushed for 2,000 or more yards in a season.

 These seven rushing seasons rank as the highest single-season rushing totals in NFL history, and reaching the 2,000-yard mark is considered a significant
achievement for running backs.

 No running back has yet achieved this feat twice.

 The first 2,000-yard season was recorded in 1973 by Buffalo Bills running back O.J. Simpson.

 He is the only player to have surpassed 2,000 yards in a 14-game season, as all others occurred in 16-game seasons; he finished the season with 2,003 rushing
yards, averaging six yards per carry and an NFL-record 143.1 rushing yards per game.

 Los Angeles Rams running back Eric Dickerson, who had broken the single-season rookie rushing record in 1983, recorded the second 2,000-yard season in 1984.

 Dickerson rushed for 2,105 yards, the current NFL rushing record, and averaged 131.6 rushing yards per game.
</item 7>

<item 8>
Marc Mueller
Marc Mueller (born April 15, 1989) is the current running backs coach of the Calgary Stampeders of the Canadian Football League (CFL).

 In college, he played quarterback for the Regina Rams of the CIS.

 He signed as an undrafted free agent with the Edmonton Eskimos of the CFL on May 9, 2011 following the 2011 CFL Draft.

 He was released on June 20, 2011 in order to play his final year for the Rams.

 Mueller is the grandson of former CFL quarterback and coach Ron Lancaster.
</item 8>

<item 9>
Richard Harris (American football)
Richard Drew Harris (January 21, 1948 – July 26, 2011) was an American football defensive end who played seven seasons in the National Football League.

 He was and All-American in 1970 for Grambling and was drafted in the first round (5th overall pick) of the 1971 NFL Draft by the Philadelphia Eagles.

 He was named All-Rookie in 1971, playing defensive end.

 Harris spent seven seasons as a defensive end in the NFL, the first three with the Philadelphia Eagles, the next two with the Chicago Bears and the final two
with the Seattle Seahawks.
</item 9>

<item 10>
Dolph Pulliam
Adolphus "Dolph" Pulliam (born March 14, 1946 in West Point, Mississippi) is a former American basketball player.

 He played collegiately at Drake University.

 He was named to Drake University's All-Century basketball team.

 He helped lead the Drake men's basketball team to the 1969 Final Four and a third-place finish.

 On 7 February 2009, Drake University retired Pulliam's #5 jersey that he wore for the Bulldogs.
</item 10>
=== END ITEM id=9fbdc9575af8f321 ===

=== ITEM id=bf5c01aab9b66028 ===
Gold sentences and facts:
Sentence 0:  It is a companion book to "A Wrinkle in Time", and part of the Time Quintet.
  Fact 0.0: The book is a companion book to 'A Wrinkle in Time'.
  Fact 0.1: The book is part of the Time Quintet.
Sentence 1: The Time Quintet is a fantasy/science fiction series of five young adult novels written by Madeleine L'Engle.
  Fact 1.0: The Time Quintet is a fantasy/science fiction series.
  Fact 1.1: The Time Quintet is a series of five young adult novels.
  Fact 1.2: The Time Quintet was written by Madeleine L'Engle.

Retrieved items:
<item 1>
Gate of Ivory, Gate of Horn
Gate of Ivory, Gate of Horn is a fantasy novel by British author Robert Holdstock.

 It was originally published in the United States in 1997 (and in the United Kingdom under the title "Gate of Ivory" in 1998.)

 The story is a prequel to "Mythago Wood" and explores Christian Huxley's quest into Ryhope Wood and the apparent suicide of his mother, Jennifer Huxley.

 The title of the book refers to the gates of horn and ivory described in both Homer's "Odyssey" and Virgil's "Aeneid".
</item 1>

<item 2>
Dust in the Wind (film)
Dust in the Wind is a 1986 film by Taiwanese filmmaker Hou Hsiao-hsien.

 It is based on co-screenwriter Wu Nien-jen's own experiences, and is the final part of Hou's coming-of-age trilogy, with the others being "A Summer at
Grandpa's" (1984) and "The Time to Live and the Time to Die" (1985).
</item 2>

<item 3>
John Howard (author)
John Howard is an English author, born in London in 1961.

 His fiction has appeared in anthologies, magazines, and the collections "The Silver Voices", "Written by Daylight", and "Cities and Thrones and Powers".

 The majority of Howard's stories have central and eastern European settings; many are set in the fictional Romanian town of Steaua de Munte.

 "The Defeat of Grief" is a novella set in Steaua de Munte and the real Black Sea resort of Balcic; the novellas "The Fatal Vision" (in "Cities and Thrones and
Powers") and "The Lustre of Time" form part of an ongoing series with Steaua de Munte architect and academic Cristian Luca as protagonist.

 "Numbered as Sand or the Stars" attempts a 'secret history' of Hungary between the World Wars.
</item 3>

<item 4>
The Four Winds of Love
The Four Winds of Love is the overall title for a series of six novels written by Compton Mackenzie, "The East Wind of Love" (1937), "The South Wind of Love"
(1938), "The West Wind of Love" (1940), "West to North" (1942), "The North Wind of Love, Book 1" (1944) and "The North Wind of Love, Book 2" (1945), which taken
together constitute a major fictional chronicle of the first forty years of the twentieth century.

 The main protagonist of the hexalogy is the semi-autobiographical character of John Ogilvie.
</item 4>

<item 5>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 5>

<item 6>
The Happy Journey to Trenton and Camden
The Happy Journey to Trenton and Camden is a one act play by American novelist and playwright Thornton Wilder written in 1931.

 It was first published in "The Long Christmas Dinner and Other Plays in One Act" (New York: Coward-McCann, 1931).
</item 6>

<item 7>
Edgar Allan Poe’s humorous “A Predicament,” originally “The Scythe of Time,” parodies Gothic sensation tales and is often paired with “How to Write a Blackwood
Article.” British writer Michael Moorcock (born 1939), famed for Elric, wrote the science-fiction novels *Behold the Man* (expanded from a 1966 novella; title
from John’s Gospel), *The Final Programme*, *The Warlord of the Air* (1971; an
</item 7>

<item 8>
A Wind in the Door
A Wind in the Door is a young adult science fantasy novel by Madeleine L'Engle.

 It is a companion book to "A Wrinkle in Time", and part of the Time Quintet.
</item 8>

<item 9>
Time Quintet
The Time Quintet is a fantasy/science fiction series of five young adult novels written by Madeleine L'Engle.
</item 9>

<item 10>
Warlord of the Air
The Warlord of the Air is a 1971 British alternate history science fiction novel written by Michael Moorcock.

 It concerns the adventures of Oswald Bastable, an Edwardian era soldier stationed in India, and his adventures in an alternate universe, in his own future,
wherein the First World War never happened.

 It is the first part of Moorcock's 'A Nomad of the Time Streams' trilogy and, in its use of speculative technology (such as airships) juxtaposed against an
Edwardian setting, it is widely considered to be one of the first steampunk novels.

 The novel was first published by Ace Books as part of their Ace Science Fiction Specials series.
</item 10>
=== END ITEM id=bf5c01aab9b66028 ===

=== ITEM id=f5eb4bef31712037 ===
Gold sentences and facts:
Sentence 0:  There is no optimal strategy involved as play entirely rests upon a random drawing of the cards.
  Fact 0.0: There is no optimal strategy involved in the game.
  Fact 0.1: Play in the game entirely rests upon a random drawing of the cards.
Sentence 1: Diplomacy is a strategic board game created by Allan B. Calhamer in 1954 and released commercially in 1959.
  Fact 1.0: Diplomacy is a strategic board game.
  Fact 1.1: Diplomacy was created by Allan B. Calhamer.
  Fact 1.2: Diplomacy was created in 1954.
  Fact 1.3: Diplomacy was released commercially in 1959.
Sentence 2:  Its main distinctions from most board wargames are its negotiation phases (players spend much of their time forming and betraying alliances with other players and forming beneficial strategies) and the absence of dice and other game elements that produce random effects.
  Fact 2.0: Diplomacy's main distinctions from most board wargames include its negotiation phases.
  Fact 2.1: In Diplomacy's negotiation phases, players spend much of their time forming and betraying alliances with other players.
  Fact 2.2: In Diplomacy's negotiation phases, players spend much of their time forming beneficial strategies.
  Fact 2.3: Diplomacy's main distinctions from most board wargames include the absence of dice and other game elements that produce random effects.

Retrieved items:
<item 1>
Kolejka (game)
Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National
Remembrance.

 It was designed to teach young generations about the hardships under communism (1945–1989), particularly the difficulty of acquiring consumer goods in the
shortage economy.

 The game has been described as inspired by "Monopoly".
</item 1>

<item 2>
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 2>

<item 3>
Sher-bakar
Sher-bakar is a two-player abstract strategy board game from Punjab, India.

 It is a hunt game.

 It uses an Alquerque board, and therefore, Sher-bakar is specifically a tiger hunt game (or tiger game).

 There are two tigers attempting to elude and capture as many of the other player's pieces which in other hunt games in this part of the world is often referred
to as a goat, cows, lamb, or men.

 An interesting and uncommon feature in this game is that the goats, cows, lamb, or men are piled up on four points of the board at the beginning of the game.

 Piling up pieces is an unusual feature in hunt games or any board game in general.

 The only other hunt game that uses this feature is Bagh bandi, a game closely related to Sher-bakar.

 Hereinforth, the white pieces will be referred to as goats.
</item 3>

<item 4>
Simultaneous action selection
Simultaneous action selection, or SAS, is a game mechanic that occurs when players of a game take action (such as moving their pieces) at the same time.

 An example of a game that uses this type of movement is the game Diplomacy.

 Typically, a "secret yet binding" method of committing to one's move is necessary, so that as players' moves are revealed and implemented, others do not change
their moves in light of the new information.

 Thus, in Diplomacy, players write down their moves and then reveal them simultaneously.

 Because no player gets the first move, this potentially arbitrary source of advantage is not present.

 It is also possible for simultaneous movement games to proceed relatively quickly, because players are acting at the same time, rather than waiting for their
turn.

 Simultaneous action selection is easily implemented in card games such as Apples to Apples in which players simply select cards and throw them face-down into
the center.
</item 4>

<item 5>
Diplomacy (game)
Diplomacy is a strategic board game created by Allan B. Calhamer in 1954 and released commercially in 1959.

 Its main distinctions from most board wargames are its negotiation phases (players spend much of their time forming and betraying alliances with other players
and forming beneficial strategies) and the absence of dice and other game elements that produce random effects.

 Set in Europe before the beginning of World War I, "Diplomacy" is played by two to seven players, each controlling the armed forces of a major European power
(or, with fewer players, multiple powers).

 Each player aims to move his or her few starting units and defeat those of others to win possession of a majority of strategic cities and provinces marked as
"supply centers" on the map; these supply centers allow players who control them to produce more units.
</item 5>

<item 6>
Alhambra (board game)
Alhambra (German: Der Palast von Alhambra , literally "The Palace of the Alhambra") is a 2003 tile-based German-style board game designed by Dirk Henn.

 It was originally published in Germany by Queen Games in a language-interdependent version; an English-specific version was released in North America by the
now-defunct Überplay.

 The game is an Arabian-themed update, set during the construction of the Alhambra palace in 14th century Granada, of the 1998 stock trading board game "Stimmt
So!"

, which in turn was an update of the 1992 mafia influence board game "Al Capone"; the original version was subsequently released as "".
</item 6>

<item 7>
Uncle Wiggily
Uncle Wiggily Longears is the main character of a series of children's stories by American author Howard R. Garis.

 He began writing the stories for the "Newark News" in 1910.

 Garis penned an Uncle Wiggily story every day (except Sundays) for more than 30 years, and published 79 books within the author's lifetime.

 According to his obituary in the "Chicago Tribune", a walk in the woods in Verona, New Jersey was his inspiration.

 The books featured work by several illustrators, notably Lansing Campbell.

 Other illustrators of the series included George L. Carlson, Louis Wisa, Elmer Rache, Edward Bloomfield, Lang Campbell and Mary and Wallace Stover.
</item 7>

<item 8>
Uncle Wiggily (board game)
Uncle Wiggily Game is a track board game based on a character in a series of children's books by American writer Howard Roger Garis.

 The game is of the "racing" variety in the style of the European "Goose Game".

 Players advance along the track from Uncle Wiggily's Bungalow to Dr. Possum's House.

 There is no optimal strategy involved as play entirely rests upon a random drawing of the cards.

 The game was first published by Milton Bradley in 1916 and has seen several editions with minor modifications over the years.

 "Uncle Wiggily" remains one of the first and favorite games of childhood, and, with "Candy Land", is considered a classic juvenile American board game.
</item 8>

<item 9>
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
</item 9>

<item 10>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 10>
=== END ITEM id=f5eb4bef31712037 ===

=== ITEM id=2c5b3bca59327e34 ===
Gold sentences and facts:
Sentence 0:  Obertraun is a popular holiday destination offering activities such as skiing, snowboarding in the winter and mountain biking, swimming and kayaking in the summer.
  Fact 0.0: Obertraun is a popular holiday destination.
  Fact 0.1: Obertraun offers skiing and snowboarding in the winter.
  Fact 0.2: Obertraun offers mountain biking in the summer.
  Fact 0.3: Obertraun offers swimming and kayaking in the summer.
Sentence 1:  The name "Salzkammergut" literally means "Estate of the Salt Chamber" and derives from the Imperial Salt Chamber, the authority charged with running the precious salt mines of the Habsburg Monarchy.
  Fact 1.0: The name 'Salzkammergut' literally means 'Estate of the Salt Chamber'.
  Fact 1.1: The name 'Salzkammergut' derives from the Imperial Salt Chamber.
  Fact 1.2: The Imperial Salt Chamber was the authority charged with running the precious salt mines of the Habsburg Monarchy.

Retrieved items:
<item 1>
Aspen/Snowmass
Aspen Snowmass is a winter resort complex located in Pitkin County in western Colorado in the United States.

 Owned and operated by the Aspen Skiing Company it comprises four skiing/snowboarding areas on four adjacent mountains in the vicinity of the towns of Aspen and
Snowmass Village.

 The four areas collectively form one of the most famous winter resorts in the world and are annually the destination for visitors from all over the world.
</item 1>

<item 2>
Salzkammergut
The Salzkammergut is a resort area located in Austria.

 It stretches from the City of Salzburg eastwards along the Austrian Alpine Foreland and the Northern Limestone Alps to the peaks of the Dachstein Mountains,
spanning the federal states of Upper Austria, Salzburg, and Styria.

 The main river of the region is the Traun, a right tributary of the Danube.

 The name "Salzkammergut" literally means "Estate of the Salt Chamber" and derives from the Imperial Salt Chamber, the authority charged with running the
precious salt mines of the Habsburg Monarchy.

 It is a UNESCO World Heritage Site.
</item 2>

<item 3>
Upper Saxon German
Upper Saxon (German: "Obersächsisch" ) is an East Central German dialect spoken in much of the modern German State of Saxony and in the adjacent parts of
Saxony-Anhalt and Thuringia.

 Though colloquially called "Saxon" ("Sächsisch" ), it is not to be confused with the Low Saxon dialect group in Northern Germany.

 Upper Saxon is closely linked to the Thuringian dialect spoken in the adjacent areas to the west.
</item 3>

<item 4>
Soviet Unterzoegersdorf
Soviet Unterzoegersdorf (German: "Sowjet-Unterzögersdorf" ) is a fictitious country created by the art/technology/theory group monochrom.

 It is the "last existing appanage republic of the USSR", located inside the Republic of Austria.

 Unterzögersdorf is partially based on an existing village well-known to Johannes Grenzfurthner, the creator of the concept.

 As a kid and teenager, Grenzfurthner spent a lot of time at his grandparents' farm in the small village of Unterzögersdorf (a cadastral municipality of
Stockerau).

 His grandparents' stories about the Naziism, WWII and the Soviet occupation in allied-occupied Austria (1945-1955) form the inspirational basis of the
long-term project "Soviet Unterzoegersdorf".
</item 4>

<item 5>
Obertraun
Obertraun, Upper Austria is a village in the Salzkammergut, a region in Austria.

 It is located near the Hallstätter See (Hallstatt Lake) and Hoher Dachstein.

 Obertraun is a popular holiday destination offering activities such as skiing, snowboarding in the winter and mountain biking, swimming and kayaking in the
summer.
</item 5>

<item 6>
Hundsheimer Berg
Hundsheimer Berg is a hill located in the Hundsheimer Berge hill range in Lower Austria, Austria close to the border with Slovakia.

 Its peak is 481 m above sea level which makes it the highest hill in the Hundsheimer Berge.

 The north slopes of the Hundsheimer Berg are covered with forests while the southern slopes have a steppe vegetation.

 Since 1965, a nature reserve with the same name, "Hundsheimer Berg", is located on these southern slopes.
</item 6>

<item 7>
Appenzeller cheese
Appenzeller cheese is a hard cow's-milk cheese produced in the Appenzell region of northeast Switzerland.

 A herbal brine, sometimes incorporating wine or cider, is applied to the wheels of cheese while they cure, which flavors and preserves the cheese while
promoting the formation of a rind.
</item 7>

<item 8>
Appenzell Alps
The Appenzell Alps (German: "Appenzeller Alpen" ) are a mountain range in Switzerland on the northern edge of the Alps.

 They extend into the cantons of Appenzell Ausserrhoden, Appenzell Innerrhoden and St. Gallen and are bordered by the Glarus Alps to the west and the Rätikon to
the south-east.
</item 8>

<item 9>
List of Austrian mountain climbers
This is a list of Austrian mountaineers.

 On 5 of the 14 Eight-thousanders Austrians have made the first ascent, more than any other nation can claim.

 Out of the total of 9 Austrian mountaineers who made first ascents of Eight-thousanders, 3 have been members of the Edelweiss Club Salzburg, an association of
mountaineers founded in Salzburg in 1881
</item 9>

<item 10>
Graph query row:
Salzkammergut: Salzkammergut
relationship: LOCATED_IN
related_entity: Obertraun
connection_path: ['Salzkammergut', 'Upper Austria', 'Obertraun']
path_relationships: ['LOCATED_IN', 'LOCATED_IN']
</item 10>
=== END ITEM id=2c5b3bca59327e34 ===

=== ITEM id=05295efd1a0f81fa ===
Gold sentences and facts:
Sentence 0: Javier Alberto Frana (born 25 December 1966) is a former tennis player from Argentina and current tennis commentator for ESPN Latin America.
  Fact 0.0: Javier Alberto Frana was born on 25 December 1966.
  Fact 0.1: Javier Alberto Frana is a former tennis player.
  Fact 0.2: Javier Alberto Frana is from Argentina.
  Fact 0.3: Javier Alberto Frana is a current tennis commentator.
  Fact 0.4: Javier Alberto Frana is a tennis commentator for ESPN Latin America.
Sentence 1: Thomaz Koch (born May 11, 1945 in Porto Alegre), is a left-handed former tennis player from Brazil, who was a quarter-finalist at the French Open, Wimbledon and the U.S. National Championships.
  Fact 1.0: Thomaz Koch was born on May 11, 1945.
  Fact 1.1: Thomaz Koch was born in Porto Alegre.
  Fact 1.2: Thomaz Koch is a left-handed former tennis player.
  Fact 1.3: Thomaz Koch is from Brazil.
  Fact 1.4: Thomaz Koch was a quarter-finalist at the French Open.
  Fact 1.5: Thomaz Koch was a quarter-finalist at Wimbledon.
  Fact 1.6: Thomaz Koch was a quarter-finalist at the U.S. National Championships.

Retrieved items:
<item 1>
1996 Nottingham Open – Singles
Javier Frana was the defending champion but lost in the first round to Marc-Kevin Goellner.
</item 1>

<item 2>
Javier Frana
Javier Alberto Frana (born 25 December 1966) is a former tennis player from Argentina and current tennis commentator for ESPN Latin America.
</item 2>

<item 3>
1993 ATP Buenos Aires – Doubles
Pablo Albano and Javier Frana were the defending champions, but lost in the quarterfinals this year.
</item 3>

<item 4>
1995 Nottingham Open – Singles
Javier Frana defeated Todd Woodbridge 7–6, 6–3 in the final to secure the title.
</item 4>

<item 5>
Thomaz Koch
Thomaz Koch (born May 11, 1945 in Porto Alegre), is a left-handed former tennis player from Brazil, who was a quarter-finalist at the French Open, Wimbledon and
the U.S. National Championships.
</item 5>

<item 6>
Paulo Borrachinha
Paulo Henrique Costa (born April 21, 1991), better known as Paulo Borrachinha, is a Brazilian mixed martial artist currently competing in the Middleweight
division of the Ultimate Fighting Championship.

 In the day August 3 of 2017, Borracinha and the Clube Atlético Mineiro officialized institutional partnership.

 The agreement did not create a bond between the club and the athlete.

 The agreement allows the use of the athletic shield of Atlético in the training uniforms of the fighter, within the possibilities of his contract with the UFC.
</item 6>

<item 7>
1994 Movistar Open – Singles
Javier Frana was the defending champion but lost in the quarterfinals to Alberto Berasategui.
</item 7>

<item 8>
1996 Abierto Mexicano de Tenis – Doubles
Javier Frana and Leonardo Lavalle were the defending champions but lost in the first round to Luis Lobo and Javier Sánchez.
</item 8>

<item 9>
Franz Josef Och
Franz Josef Och (November 2, 1971) is a German computer scientist.

 He is currently working at Grail (an Illumina company) as Head of Data Science.

 Prior to this he was Chief Data Scientist at Human Longevity Inc., and earlier worked for Google as a Distinguished Research Scientist and head of machine
translation based at Google's Mountain View, California, headquarters south of San Francisco.
</item 9>

<item 10>
The material compiles tennis results and player biographies. Sherwood Stewart and Ferdi Taygan won the 1982 French Open doubles and finished runner-up at the
Volvo Masters and U.S. Pro Indoor; Stewart later partnered others. Jorge Lozano, a Mexican doubles No. 4, won nine doubles titles, including two French Open
mixed doubles crowns, and captained Mexico’s Davis Cup team. Brazilian Carlos Fernandes reached the 1961 French Open quarterfinals. Argentine Javier Fr
</item 10>
=== END ITEM id=05295efd1a0f81fa ===

=== ITEM id=1527706d0d9f05f6 ===
Gold sentences and facts:
Sentence 0: "I'll Keep the Lovelight Burning" is a popular song written in 1942 by Harry Tobias, Nick Kenny, and Harold Levey, popularized by Patti Page in 1949.
  Fact 0.0: "I'll Keep the Lovelight Burning" is a popular song.
  Fact 0.1: "I'll Keep the Lovelight Burning" was written in 1942.
  Fact 0.2: "I'll Keep the Lovelight Burning" was written by Harry Tobias.
  Fact 0.3: "I'll Keep the Lovelight Burning" was written by Nick Kenny.
  Fact 0.4: "I'll Keep the Lovelight Burning" was written by Harold Levey.
  Fact 0.5: "I'll Keep the Lovelight Burning" was popularized by Patti Page.
  Fact 0.6: Patti Page popularized "I'll Keep the Lovelight Burning" in 1949.
Sentence 1: Clara Ann Fowler (November 8, 1927 – January 1, 2013), known by her professional name Patti Page, was an American singer of traditional pop music and country music.
  Fact 1.0: Clara Ann Fowler's date range is November 8, 1927 to January 1, 2013.
  Fact 1.1: Clara Ann Fowler is known by the professional name Patti Page.
  Fact 1.2: Patti Page was an American singer.
  Fact 1.3: Patti Page was a singer of traditional pop music.
  Fact 1.4: Patti Page was a singer of country music.

Retrieved items:
<item 1>
Elton John
Sir Elton Hercules John, (born Reginald Kenneth Dwight; 25 March 1947), is an English singer, pianist, and composer.

 He has worked with lyricist Bernie Taupin as his songwriting partner since 1967; they have collaborated on more than 30 albums to date.

 In his five-decade career Elton John has sold more than 300million records, making him one of the best-selling music artists in the world.

 He has more than fifty Top 40 hits, including seven consecutive No. 1 US albums, 58 "Billboard" Top 40 singles, 27 Top 10, four No. 2 and nine No. 1.

 For 31 consecutive years (1970–2000) he had at least one song in the "Billboard" Hot 100.

 His tribute single, re-penned in dedication to the late Princess Diana, "Candle in the Wind 1997" sold over 33million copies worldwide and is the best-selling
single in the history of the UK and US singles charts.

 He has also composed music, produced records, and has occasionally acted in films.
</item 1>

<item 2>
Albert Sharpe
Albert Sharpe (15 April 1885 – 13 February 1970) was an Irish stage and film actor.

 His most famous roles were those of Darby O'Gill in Disney's "Darby O'Gill and the Little People" and as Finian McLonergan in the Original Broadway production
of the musical "Finian's Rainbow".

 (the film version, made in 1968, had Fred Astaire in the role.)

 On screen he played Fiona's father Andrew in the MGM musical Brigadoon.

 He was also a member of the Abbey Players.

 His last ten years were spent in retirement.

 He died in 1970 in Belfast at the age of 84.
</item 2>

<item 3>
For You (Ricky Nelson song)
"For You" is a song written by Joe Burke and Al Dubin in 1930.

 Dubin wrote the songs for the Broadway show "42nd Street".

 The Glen Gray Orchestra recorded it with Kenny Sargent doing the vocals.

 Perry Como recorded it in November 1947, releasing the song in 1948.

 Dean Martin recorded it for an album in 1957.

 Timi Yuro also recorded the song.

 It was released on her LP "Timi Yuro" (Liberty LRP-3208 (mono) and Liberty LST-7208 (stereo)) in 1961.
</item 3>

<item 4>
Keep This Fire Burning
"Keep This Fire Burning" is a song by Swedish pop singer Robyn, released as the first single from her third album "Don't Stop the Music".

 It was released in Sweden on September 21, 2002, where it became her highest charting single since 1995's "Do You Really Want Me (Show Respect)".

 The song was also released as a single in Australia under the name "By Your Side", due to the Australian bush fires which were happening at the time.

 In 2008, a re-recorded version of the song appeared on the special edition of Robyn's eponymous album.
</item 4>

<item 5>
Tony Hiller
Anthony Toby "Tony" Hiller (born 30 July 1927) is a British songwriter.

 He began his musical career as a member of the song and dance duo The Hiller Brothers, sharing the stage with his brother Irving.

 The Hiller Brothers appeared with many artists of the time including Alma Cogan, Tommy Cooper, Val Doonican, Matt Monro, The Shadows, Bernard Manning, Kathy
Kirby, Roger Whittaker, Rip Taylor, Gene Vincent, Lance Percival, Tessie O'Shea, Frank Ifield, Deep River Boys, The Dallas Boys, Clark Brothers, Paul Melba and
Ray Burns.
</item 5>

<item 6>
Betty Ann Davies
Betty Ann Davies (24 December 1910 – 14 May 1955) was a British stage and film actress active from the 1920s to the 1950s.

 She died young in 1955 aged 44 following an operation.

 Miss Davies made her first stage appearance at the Palladium in a revue in 1924.

 The following year she joined Cochran's Young Ladies in revues such as "One Dam Thing After Another" and "This Year of Grace".

 Betty Ann Davies enjoyed a long and distinguished West End career which included "The Good Companions" 1934, "Morning Star" 1942, and "Blithe Spirit" 1943.

 Her outstanding stage triumph was in the role of Blanche du Bois, which she took over from Vivien Leigh, in the original West End production of " A Streetcar
Named Desire".

 Miss Davis appeared in 38 films, and was active in TV at the time of her death.

 She left one son Brook Blackford.
</item 6>

<item 7>
Willie and the Hand Jive
"Willie and the Hand Jive" is a song written by Johnny Otis and originally released as a single in 1958 by Johnny Otis Show, reaching #9 on the "Billboard" Hot
100 chart and #5 on the Billboard R&B chart.

 The song has a Bo Diddley beat and was partly inspired by the music sung by a chain gang Otis heard while he was touring.

 The lyrics are about a man who became famous for doing a dance with his hands, but the song has been accused of glorifying masturbation.

 It has since been covered by numerous artists, including The Strangeloves, Eric Clapton, Cliff Richard, Kim Carnes, George Thorogood and The Grateful Dead.

 Clapton's 1974 version was also released as a single and also reached the Billboard Top 40, peaking at #26.

 Thorogood's 1985 version reached #25 on the Hot Mainstream Rock Tracks chart.
</item 7>

<item 8>
I'll Keep the Lovelight Burning
"I'll Keep the Lovelight Burning" is a popular song written in 1942 by Harry Tobias, Nick Kenny, and Harold Levey, popularized by Patti Page in 1949.

 Louis Armstrong also covered the song in 1949.
</item 8>

<item 9>
You Keep Me Hangin' On
"You Keep Me Hangin' On" is a 1966 song written and composed by Holland–Dozier–Holland.

 It first became a popular "Billboard" Hot 100 number one hit for the American Motown group The Supremes in late 1966.

 The rock band Vanilla Fudge covered the song a year later and had a top ten hit with their version.

 British pop singer Kim Wilde covered "You Keep Me Hangin' On" in 1986, bumping it back to number one on the "Billboard" Hot 100 in June 1987.

 The single reached number one by two different musical acts in America.

 In the first 32 years of the "Billboard" Hot 100 rock era, “You Keep Me Hangin' On” became one of only six songs to achieve this feat.

 In 1996, country music singer Reba McEntire's version reached number 2 on the US "Billboard" Hot Dance Club Play chart.
</item 9>

<item 10>
Patti Page
Clara Ann Fowler (November 8, 1927 – January 1, 2013), known by her professional name Patti Page, was an American singer of traditional pop music and country
music.

 She was the top-charting female vocalist and best-selling female artist of the 1950s, selling over 100 million records during a six-decade long career.

 She was often introduced as "the Singin' Rage, Miss Patti Page".

 New York WNEW disc-jockey William B. Williams introduced her as "A Page in my life called Patti".
</item 10>
=== END ITEM id=1527706d0d9f05f6 ===

=== ITEM id=ac014ea5de2068d3 ===
Gold sentences and facts:
Sentence 0: Cantarella was a poison allegedly used by the Borgias during the papacy of Pope Alexander VI.
  Fact 0.0: Cantarella was a poison
  Fact 0.1: Cantarella was allegedly used by the Borgias
  Fact 0.2: The alleged use of Cantarella by the Borgias occurred during the papacy of Pope Alexander VI
Sentence 1:  The use of this poison is not well documented in any of the papal records and it was most likely conceived after 1503 as part of Pope Julius II's effort to remove his name from the records.
  Fact 1.0: The use of Cantarella is not well documented in any of the papal records
  Fact 1.1: Cantarella was most likely conceived after 1503
  Fact 1.2: Cantarella was most likely conceived as part of Pope Julius II's effort to remove his name from the records
Sentence 2: Damnatio memoriae is a modern Latin phrase literally meaning "condemnation of memory", meaning that a person must not be remembered.
  Fact 2.0: Damnatio memoriae is a modern Latin phrase
  Fact 2.1: Damnatio memoriae literally means condemnation of memory
  Fact 2.2: Damnatio memoriae means that a person must not be remembered

Retrieved items:
<item 1>
Dum Diversas
Dum Diversas (English: "Until different") is a papal bull issued on 18 June 1452 by Pope Nicholas V.

 It authorized Afonso V of Portugal to conquer Saracens and pagans and consign them to "perpetual servitude".

 Pope Calixtus III reiterated the bull in 1456 with "Inter Caetera" (not to be confused with Alexander VI's), renewed by Pope Sixtus IV in 1481 and Pope Leo X
in 1514 with "Precelse denotionis".

 The concept of the consignment of exclusive spheres of influence to certain nation states was extended to the Americas in 1493 by Pope Alexander VI with "Inter
caetera".
</item 1>

<item 2>
Giovanni Borgia (Infans Romanus)
Giovanni Borgia (March 1498 – 1548), known as the Infans Romanus ("the Roman child"), was born into the House of Borgia in secret and is of unclear parentage.

 Speculations of the child's parentage involve either Lucrezia Borgia with her alleged lover, Perotto Calderon or Cesare Borgia, or Pope Alexander VI as his
father.

 Cesare Borgia's biographer Rafael Sabatini says that the truth is fairly clear: Alexander fathered the child with an unknown Roman woman.
</item 2>

<item 3>
Damnatio memoriae
Damnatio memoriae is a modern Latin phrase literally meaning "condemnation of memory", meaning that a person must not be remembered.
</item 3>

<item 4>
Portrait of Pope Julius II
Portrait of Pope Julius II is an oil painting of 1511–12 by the Italian High Renaissance painter Raphael.

 The portrait of Pope Julius II was unusual for its time and would carry a long influence on papal portraiture.

 From early in its life, it was specially hung at the pillars of the church of Santa Maria del Popolo, on the main route from the north into Rome, on feast and
high holy days.

 Giorgio Vasari, writing long after Julius' death, said that "it was so lifelike and true it frightened everyone who saw it, as if it were the living man
himself".
</item 4>

<item 5>
Papal conclave, September 1503
The papal conclave of September 1503 elected Pope Pius III to succeed Pope Alexander VI.

 Due to the Italian Wars, the College of Cardinals was surrounded by three potentially hostile armies, loyal to Louis XII of France, Ferdinand II of Aragon, and
Cesare Borgia (the cardinal-nephew of Alexander VI).
</item 5>

<item 6>
Sicut Dudum
Sicut Dudum (English: "Just As Long Ago" ) is a papal bull promulgated by Pope Eugene IV in Florence on January 13, 1435, which forbade the enslavement of local
natives in the Canary Islands who had converted or were converting to Christianity.

 "Sicut Dudum" was meant to reinforce "Creator Omnium", issued the previous year, condemning Portuguese slave raids in the Canary Islands.

 Over forty years after "Creator Omnium" and "Sicut Dudum, Pope Sixtus IV found it necessary repeat the prohibition in his papal bull "Regimini Gregis" which
threatened the excommunication of all captains or pirates who enslaved Christians.
</item 6>

<item 7>
Dictatus papae
Dictatus papae is a compilation of 27 statements of powers arrogated to the Pope that was included in Pope Gregory VII's register under the year 1075.

 Some historians argue that it was written (or dictated) by Gregory VII himself; others argue that it has been inserted in the register at a later date, and
that it had a different origin.

 In 1087 Cardinal Deusdedit published a collection of decretals, dedicated to Pope Victor III, that embodied the law of the Church – Canon law – which he had
compiled from many sources, both legitimate and false (see Pseudo-Isidore).

 The "Dictatus papae" agrees so clearly and closely with this collection that some have argued the "Dictatus" must have been based on it; and so must be of a
later date of compilation and insertion in the papal register than 1087.
</item 7>

<item 8>
Cantarella
Cantarella was a poison allegedly used by the Borgias during the papacy of Pope Alexander VI.

 It was probably a variation of arsenic or cantharidin powder (made from blister beetles) .

 The use of this poison is not well documented in any of the papal records and it was most likely conceived after 1503 as part of Pope Julius II's effort to
remove his name from the records.
</item 8>

<item 9>
Pedro Luis de Borja Lanzol de Romaní
Pedro Luis de Borja Lanzol de Romaní, O.S.Io.Hieros.

 (1472–1511) was a Roman Catholic cardinal and cardinal-nephew and papal military leader.

 He received a wide variety of sinecures during the papacy of his great-uncle, Pope Alexander VI, but was exiled to Naples on the election of Borja rival Pope
Julius II.

 Borja also fought with the Knights Hospitaller in Jerusalem and Rhodes.
</item 9>

<item 10>
Pope Julius II
Pope Julius II (Italian: "Papa Giulio II" ; Latin: "Iulius II" ) (5 December 1443 – 21 February 1513), born Giuliano della Rovere, and nicknamed "The Fearsome
Pope" and "The Warrior Pope", was Pope from 1 November 1503 to his death in 1513.

 His papacy was marked by an active foreign policy, ambitious building projects, and patronage of the arts—he commissioned the destruction and rebuilding of St.
Peter's Basilica, and Michelangelo's decoration of the ceiling of the Sistine Chapel.

 In addition to an active military policy, he personally led troops into battle on at least two occasions, the first to expel Giovanni Bentivoglio from Bologna
(17 August 1506–23 March 1507), and the second in an attempt to recover Ferrara for the Papal States (1 September 1510–29 June 1512).
</item 10>
=== END ITEM id=ac014ea5de2068d3 ===

=== ITEM id=dd25eaa096fc623a ===
Gold sentences and facts:
Sentence 0: Doris May Lessing, CH ("née" Tayler; 22 October 1919  – 17 November 2013) was a British novelist, poet, playwright, librettist, biographer and short story writer.
  Fact 0.0: Doris May Lessing, CH, was born Doris May Tayler (née Tayler).
  Fact 0.1: Doris May Lessing was born on 22 October 1919.
  Fact 0.2: Doris May Lessing died on 17 November 2013.
  Fact 0.3: Doris May Lessing was a British novelist.
  Fact 0.4: Doris May Lessing was a British poet.
  Fact 0.5: Doris May Lessing was a British playwright.
  Fact 0.6: Doris May Lessing was a British librettist.
  Fact 0.7: Doris May Lessing was a British biographer.
  Fact 0.8: Doris May Lessing was a British short story writer.
Sentence 1: Philip Kindred Dick (December 16, 1928 – March 2, 1982) was an American writer notable for publishing works of science fiction.
  Fact 1.0: Philip Kindred Dick was born on December 16, 1928.
  Fact 1.1: Philip Kindred Dick died on March 2, 1982.
  Fact 1.2: Philip Kindred Dick was an American writer.
  Fact 1.3: Philip Kindred Dick is notable for publishing works of science fiction.

Retrieved items:
<item 1>
George Orwell bibliography
The bibliography of George Orwell includes journalism, essays, novels and non-fiction books written by the British writer Eric Blair (1903–50), either under his
own name or, more usually, under his pen name George Orwell.

 Orwell was a prolific writer on topics related to contemporary English society and literary criticism, whom the British newsweekly "The Economist" in 2008
declared "perhaps the 20th century's best chronicler of English culture."

 His non-fiction cultural and political criticism constitutes the majority of his work, but Orwell also wrote in several genres of fictional literature.
</item 1>

<item 2>
The Good Terrorist
The Good Terrorist is a 1985 political novel written by the British novelist Doris Lessing.

 It was first published in September that year by Jonathan Cape in the United Kingdom and by Alfred A. Knopf in the United States.

 The book's protagonist is the naïve drifter Alice, who squats with a group of radicals in London and is drawn into their terrorist activities.
</item 2>

<item 3>
Margaret Landon
Margaret Landon (September 7, 1903 – December 4, 1993) was an American writer best remembered for "Anna and the King of Siam", her best-selling 1944 novel of
the life of Anna Leonowens which eventually sold over a million copies and was translated into more than twenty languages.

 In 1950, Landon sold the musical play rights to Richard Rodgers and Oscar Hammerstein II, who created the musical "The King and I" from her book.

 A later work, "Never Dies the Dream", appeared in 1949.
</item 3>

<item 4>
Doris Lessing
Doris May Lessing, CH ("née" Tayler; 22 October 1919  – 17 November 2013) was a British novelist, poet, playwright, librettist, biographer and short story
writer.

 Her novels include "The Grass Is Singing" (1950), the sequence of five novels collectively called "Children of Violence" (1952–69), "The Golden Notebook"
(1962), "The Good Terrorist" (1985), and five novels collectively known as "Canopus in Argos: Archives" (1979–1983).
</item 4>

<item 5>
The Philip K. Dick Reader
The Philip K. Dick Reader is a collection of science fiction stories by American writer Philip K. Dick.

 It was first published by Citadel Twilight in 1997.

 Many of the stories had originally appeared in the magazines "If", "Science Fiction Adventures", "Science Fiction Stories", "Orbit", "Fantasy and Science
Fiction", "Imagination", "Future", "Galaxy Science Fiction", "Beyond Fantasy Fiction", "Satellite", "Imaginative Tales", "Fantastic Universe" and "Space Science
Fiction".

 It is identical in content and order to the edition of volume 3 of the Collected Stories of Philip K. Dick produced by the same publisher apart from the
substitution of three stories in positions 21-23 of 24 and the omission of the end notes in the Collected Stories edition.

 At press time, stories 21 and 24 had already been made into successful movie adaptations and stories 22 and 23 had been optioned.
</item 5>

<item 6>
Jenny Diski
Jenny Diski FRSL (née Simmonds; 8 July 1947 – 28 April 2016) was an English writer.

 She had a troubled childhood, but was rescued by the older novelist Doris Lessing; she lived in Lessing's house for four years.

 Diski was educated at University College London, and worked as a teacher during the 1970s and early 1980s.
</item 6>

<item 7>
Adjustment Team
"Adjustment Team" is a science fiction short story by American writer Philip K. Dick.

 It was first published in "Orbit Science Fiction" (September–October 1954, No. 4) with illustration by Faragasso.

 It was later reprinted in "The Sands of Mars and Other Stories" (Australian) in 1958, "The Book of Philip K. Dick" in 1973, "The Turning Wheel and Other
Stories" (United Kingdom) in 1977, "The Collected Stories of Philip K. Dick" in 1987 (Underwood–Miller), 1988 (Gollancz, United Kingdom), 1990 (Citadel
Twilight, United States), "Selected Stories of Philip K. Dick" in 2002 and in "The Early Work of Philip K. Dick, Volume One: The Variable Man & Other Stories"
in 2009.
</item 7>

<item 8>
Michael Herr
Michael David Herr (April 13, 1940 – June 23, 2016) was an American writer and war correspondent, known as the author of "Dispatches" (1977), a memoir of his
time as a correspondent for "Esquire magazine" (1967–1969) during the Vietnam War.

 The book was called the best "to have been written about the Vietnam War" by "The New York Times Book Review"; novelist John le Carré called it "the best book
I have ever read on men and war in our time."

 Herr later was credited with pioneering the literary genre of the nonfiction novel, along with authors such as Truman Capote, Norman Mailer, and Tom Wolfe.
</item 8>

<item 9>
Philip K. Dick
Philip Kindred Dick (December 16, 1928 – March 2, 1982) was an American writer notable for publishing works of science fiction.
</item 9>

<item 10>
Philip K. Dick Award
The Philip K. Dick Award is a science fiction award given annually at Norwescon sponsored by the Philadelphia Science Fiction Society and (since 2005) supported
by the Philip K. Dick Trust, and named after science fiction and fantasy writer Philip K. Dick.

 It has been awarded since 1983, the year after Dick's death.

 Works that have received the award are identified on their covers as "Best Original SF Paperback".

 They are awarded to the best original paperback published each year in the US.
</item 10>
=== END ITEM id=dd25eaa096fc623a ===

=== ITEM id=251d6c1047dab1aa ===
Gold sentences and facts:
Sentence 0:  Players compete in races on tracks inspired by attractions such as the Haunted Mansion, Pirates of the Caribbean, Space Mountain and Big Thunder Mountain Railroad to acquire missing parts for the park's fireworks machine, which was accidentally destroyed by Chip 'n' Dale while they were gathering acorns.
  Fact 0.0: Players compete in races.
  Fact 0.1: The race tracks are inspired by attractions such as the Haunted Mansion.
  Fact 0.2: The race tracks are inspired by attractions such as Pirates of the Caribbean.
  Fact 0.3: The race tracks are inspired by attractions such as Space Mountain.
  Fact 0.4: The race tracks are inspired by attractions such as Big Thunder Mountain Railroad.
  Fact 0.5: Players race on tracks to acquire missing parts for the park's fireworks machine.
  Fact 0.6: The park's fireworks machine was accidentally destroyed by Chip 'n' Dale.
  Fact 0.7: Chip 'n' Dale destroyed the fireworks machine while they were gathering acorns.
Sentence 1:  Opened on January 15, 1975, Space Mountain is the oldest operating roller coaster in the state of Florida, and is the original version of the iconic attraction that has since been replicated at all of The Walt Disney Company's Magic Kingdom-style theme parks worldwide, except for the Shanghai Disneyland Resort.
  Fact 1.0: Space Mountain opened on January 15, 1975.
  Fact 1.1: Space Mountain is the oldest operating roller coaster in the state of Florida.
  Fact 1.2: Space Mountain is the original version of an iconic attraction.
  Fact 1.3: The original version of Space Mountain has since been replicated at all of The Walt Disney Company's Magic Kingdom-style theme parks worldwide.
  Fact 1.4: The replication of Space Mountain excludes the Shanghai Disneyland Resort.

Retrieved items:
<item 1>
Wild West Express Coaster
Wild West Express Coaster is a steel roller coaster operating at Glenwood Caverns Adventure Park in Glenwood Springs, Colorado.

 Wild West Express Coaster opened to the public at Glenwood Caverns Adventure Park on May 25, 2012.
</item 1>

<item 2>
Space Mountain
Space Mountain is the name of a space-themed indoor roller coaster attraction located at five of the Magic Kingdom-style Disney Parks (all except for Shanghai
Disneyland Park).

 Although all five versions of the attraction are different in nature, all have a similar domed exterior façade that is a landmark for the respective park.

 The first Space Mountain ride opened in 1975 at Walt Disney World Resort in Florida, and similar attractions were then eventually built at the other Disney
parks.
</item 2>

<item 3>
Space Mountain (Disneyland)
Space Mountain (known as Hyperspace Mountain from 2015 to 2017) is an indoor, space-themed roller coaster in Tomorrowland at Disneyland in Anaheim, California.

 Opened on May 27, 1977, it was the second roller coaster built at Disneyland, and was the second of the five versions of Space Mountain built by The Walt
Disney Company.

 Its exterior façade is one of Disneyland's four "mountain" structures that serve as park landmarks.
</item 3>

<item 4>
Primeval Whirl
Primeval Whirl is a steel wild mouse roller coaster at Disney's Animal Kingdom at the Walt Disney World Resort.

 The ride is a spinning roller coaster purchased from Reverchon Industries.

 The ride is part of Chester and Hester's Dino-Rama, which is itself part of DinoLand U.S.A. It is a roller coaster in the "Mild But Wild Thrills" category.

 It has cars that spin in circles while traveling on tracks, permitting the ride experience to vary greatly each time it is ridden.

 The ride features 13 cars, each seating up to 4 riders.
</item 4>

<item 5>
Walt Disney World Dolphin
The Walt Disney World Dolphin is a resort hotel designed by architect Michael Graves located between Epcot and Disney's Hollywood Studios in the Walt Disney
World Resort in Bay Lake, Florida, next to Disney's BoardWalk Resort area.

 It opened on June 1, 1990 and is joined to its sister hotel, the Walt Disney World Swan (also designed by Graves) by a palm-tree lined covered walkway crossing
a lagoon.

 The Walt Disney World Swan and Dolphin is a joint venture between the Walt Disney Company, Tishman Hotel Corporation, MetLife and Starwood Hotels and Resorts.

 The land the resort occupies is owned by the Walt Disney Company, while the buildings themselves are leased by Disney to the Tishman Hotel Corporation and
MetLife but operated by Starwood Hotels & Resorts Worldwide under the Sheraton Hotels brand.

 The Walt Disney World Swan and Dolphin are a part of the Walt Disney Collection of resorts; because of this they are Disney branded and guests of the resort
have access to special Disney benefits available to Disney Resort Hotel guests only.
</item 5>

<item 6>
Space Mountain (Magic Kingdom)
Space Mountain is an indoor dark outer space-themed steel roller coaster at the Magic Kingdom theme park at the Walt Disney World Resort in Lake Buena Vista,
Florida.

 Opened on January 15, 1975, Space Mountain is the oldest operating roller coaster in the state of Florida, and is the original version of the iconic attraction
that has since been replicated at all of The Walt Disney Company's Magic Kingdom-style theme parks worldwide, except for the Shanghai Disneyland Resort.

 RCA helped fund Space Mountain's construction and sponsored the ride from 1975 to 1993; FedEx sponsored Space Mountain from 1994 to 2004.
</item 6>

<item 7>
Millennium Force
Millennium Force is a steel roller coaster built by Intamin at Cedar Point amusement park in Sandusky, Ohio, United States.

 It was the fourteenth roller coaster to be built at the park since Blue Streak opened in 1964.

 Upon completion in 2000, Millennium Force broke six world records and was the world's first Giga Coaster, a roller coaster that exceeds 300 ft in height and
completes a full circuit.

 It was briefly the tallest and fastest in the world until Steel Dragon 2000 opened later the same year.

 The ride is also the third-longest roller coaster in North America after The Beast at Kings Island and Fury 325 at Carowinds.

 It was the first roller coaster to use a cable lift system rather than a traditional chain lift.

 The coaster has a 310 ft , 45-degree lift hill with a 300 ft drop and features two tunnels, three overbanked turns, and four hills.

 It has a top speed of 93 mph .
</item 7>

<item 8>
Treasure Island Resort &amp; Casino
An expansion that was completed in fall 2008 included 230 new hotel rooms, 30,000-square-foot event center and a bowling center complete with an arcade area.

 In 2015, Tado Steakhouse was constructed, Tradewinds Buffet was remodeled and the water park & spa construction began.

 The Lagoon and Wave Spa opened February 9, 2016.
</item 8>

<item 9>
Rock 'n' Roller Coaster Starring Aerosmith
Rock 'n' Roller Coaster Starring Aerosmith is an enclosed launched steel roller coaster at Disney's Hollywood Studios at the Walt Disney World Resort and at the
Walt Disney Studios Park in Disneyland Paris (where it is called "Rock 'n' Roller Coaster avec Aerosmith").

 As the attraction's name suggests, the coaster features Aerosmith members, Steven Tyler, Joe Perry, Tom Hamilton, Joey Kramer, and Brad Whitford.
</item 9>

<item 10>
Walt Disney World Quest: Magical Racing Tour
Walt Disney World Quest: Magical Racing Tour is a 2000 go-kart racing video game based on attractions at the Walt Disney World Resort.

 Players compete in races on tracks inspired by attractions such as the Haunted Mansion, Pirates of the Caribbean, Space Mountain and Big Thunder Mountain
Railroad to acquire missing parts for the park's fireworks machine, which was accidentally destroyed by Chip 'n' Dale while they were gathering acorns.

 The game was developed by Crystal Dynamics and published by Eidos Interactive.

 The Game Boy Color version was developed by Prolific Publishing.
</item 10>
=== END ITEM id=251d6c1047dab1aa ===

=== ITEM id=ecdbdb6df58d5607 ===
Gold sentences and facts:
Sentence 0:  Most figures are all shaped with the body of Mickey Mouse but have different themed markings, colors, and patterns.
  Fact 0.0: Most figures are shaped with the body of Mickey Mouse.
  Fact 0.1: Most figures have different themed markings.
  Fact 0.2: Most figures have different themed colors.
  Fact 0.3: Most figures have different themed patterns.
Sentence 1:  He was created by Walt Disney and Ub Iwerks at the Walt Disney Studios in 1928.
  Fact 1.0: Mickey Mouse was created by Walt Disney.
  Fact 1.1: Mickey Mouse was created by Ub Iwerks.
  Fact 1.2: Mickey Mouse was created at the Walt Disney Studios.
  Fact 1.3: Mickey Mouse was created in 1928.

Retrieved items:
<item 1>
Donald Duck
Donald Duck is a cartoon character created in 1934 at Walt Disney Productions.

 Donald is an anthropomorphic white duck with a yellow-orange bill, legs, and feet.

 He typically wears a sailor shirt and cap with a bow tie.

 Donald is most famous for his semi-intelligible speech and his mischievous and temperamental personality.

 Along with his friend Mickey Mouse, Donald is one of the most popular Disney characters and was included in TV Guide's list of the 50 greatest cartoon
characters of all time in 2002.

 He has appeared in more films than any other Disney character, and is the most published comic book character in the world outside of the superhero genre.
</item 1>

<item 2>
Pete (Disney)
Pete (also called Peg-Leg Pete, Pistol Pete and Black Pete, among other names) is an anthropomorphic cartoon character created in 1925 by Walt Disney and Ub
Iwerks.

 He is a character of The Walt Disney Company and often appears as a nemesis and the main antagonist in Mickey Mouse universe stories.

 He was originally an anthropomorphic bear but with the advent of Mickey Mouse in 1928, he was defined as a cat.

 Pete is the oldest continuing Disney character, having debuted three years before Mickey Mouse in the cartoon "Alice Solves the Puzzle" (1925).
</item 2>

<item 3>
Get a Horse!
Get a Horse!

 is a 2013 American 3D animated slapstick comedy short film, produced by Walt Disney Animation Studios.

 Combining black-and-white hand-drawn animation and color CGI animation, the short features the characters of the late 1920s "Mickey Mouse" cartoons, and
features archival recordings of Walt Disney in his posthumous role as Mickey Mouse.

 It is the first original Mickey Mouse theatrical animated short since "Runaway Brain" (1995), and the first appearance of Oswald the Lucky Rabbit in a Disney
animated production in 85 years.
</item 3>

<item 4>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 4>

<item 5>
Vinylmation
Vinylmation is a brand of 1.5", 3", and 9" vinyl collectibles sold at Disney theme parks, select Disney Stores, and the online Disney Store.

 The name "Vinylmation" is a combination of the word "Animation" and "Vinyl".

 Most figures are all shaped with the body of Mickey Mouse but have different themed markings, colors, and patterns.

 This means that all figures have round mouse ears, whether or not the painted character has them.

 This is accommodated by painting the ears with backgrounds - stars, stripes, rainbows - to match the painted character.

 The series exception is the Park Starz series which does not resemble Mickey at all, and more closely resembles iconic figures from the parks.

 Vinylmation was first introduced in July 2008, and the first figures were introduced in November 2008.
</item 5>

<item 6>
Hagop Sandaldjian
Hagop Sandaldjian (1931–1990) was an Egyptian-born Armenian American musician and microminiature sculptor, best known for his tiny art pieces displayed at the
Museum of Jurassic Technology in Los Angeles, California.

 Sandaldjian's creations included a carving of Mount Ararat on a grain of rice; a crucifix in which a minute golden figure of Jesus hangs upon a cross made from
a bisected strand of Sandaldjian's own hair; and recreations of Disney figures (Snow White and the Seven Dwarfs or Mickey Mouse, for example) or historical
figures (such as Napoleon or Pope John Paul II) presented in the eye or on the tip of a needle.
</item 6>

<item 7>
Mickey Mouse
Mickey Mouse is a funny animal cartoon character and the official mascot of The Walt Disney Company.

 He was created by Walt Disney and Ub Iwerks at the Walt Disney Studios in 1928.

 An anthropomorphic mouse who typically wears red shorts, large yellow shoes, and white gloves, Mickey has become one of the world's most recognizable
characters.
</item 7>

<item 8>
Mickey's Christmas Carol
Mickey's Christmas Carol is a 1983 American animated featurette produced by Walt Disney Productions and released by Buena Vista Distribution.

 It was directed and produced by Burny Mattinson.

 The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.

 Many other Disney characters, primarily from the Mickey Mouse universe, "Robin Hood", and "The Adventures of Ichabod and Mr. Toad", were cast throughout the
film.
</item 8>

<item 9>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 9>

<item 10>
Bill Barretta
William Paul "Bill" Barretta (born June 19, 1964) is an American puppeteer and producer who has been performing with The Muppets since 1991, when he puppeteered
the body of Sinclair family patriarch, Earl Sinclair on "Dinosaurs".

 He later developed several new characters on "Muppets Tonight", including Pepe the King Prawn, Johnny Fiama, Big Mean Carl and Bobo the Bear.

 Along with having his own Muppet characters, Barretta has taken over several of Jim Henson's roles, such as Dr. Teeth, Rowlf the Dog, Mahna Mahna and Swedish
Chef, and briefly took over Jerry Nelson's role of Lew Zealand.

 His film debut as a principal puppeteer was in 1996's "Muppet Treasure Island" as Clueless Morgan.

 In addition, Barretta has produced two of the Muppets' television films, "It's a Very Merry Muppet Christmas Movie" (2002) and "The Muppets' Wizard of Oz"
(2005).

 Barretta also provides additional voices on "Kim Possible".
</item 10>
=== END ITEM id=ecdbdb6df58d5607 ===
