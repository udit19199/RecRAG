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

=== ITEM id=fa639c256592cc0a ===
Gold sentences and facts:
Sentence 0: Chance 's friend Pat Wheeler ( Ward Bond ) and his wagon train of supplies stop in town , with a young gunslinger , Colorado Ryan ( Ricky Nelson ) , riding guard . Inside the jail , Stumpy ( Walter Brennan ) , Chance 's game - legged deputy , keeps watch over the jail and Joe , who knows that Stumpy holds an old grudge against Joe 's wealthy and powerful brother . Joe warns his jailers that Nathan Burdette will not like how his brother is being treated .
  Fact 0.0: Pat Wheeler is Chance's friend.
  Fact 0.1: Pat Wheeler is played by Ward Bond.
  Fact 0.2: Pat Wheeler has a wagon train of supplies.
  Fact 0.3: Pat Wheeler and his wagon train of supplies stop in town.
  Fact 0.4: Colorado Ryan is a young gunslinger.
  Fact 0.5: Colorado Ryan is played by Ricky Nelson.
  Fact 0.6: Colorado Ryan is riding guard as Pat Wheeler's wagon train stops in town.
  Fact 0.7: Joe is in the jail.
  Fact 0.8: Stumpy is played by Walter Brennan.
  Fact 0.9: Stumpy is Chance's game-legged deputy.
  Fact 0.10: Stumpy keeps watch over the jail.
  Fact 0.11: Stumpy keeps watch over Joe.
  Fact 0.12: Joe knows that Stumpy holds an old grudge against Joe's wealthy and powerful brother.
  Fact 0.13: Joe's brother is wealthy and powerful.
  Fact 0.14: Joe warns his jailers.
  Fact 0.15: Joe warns his jailers that Nathan Burdette will not like how Joe, Nathan's brother, is being treated.
Sentence 1: Walter Brennan as Stumpy
  Fact 1.0: Walter Brennan plays the character Stumpy.

Retrieved items:
<item 1>
Rio Bravo (film)
Chance is ordered to take the men to the jail to let Joe go . He is reluctant to do so , but Dude implores Chance to cooperate , saying that Stumpy is alone and
has no food or water to hold out very long . The remaining Burdette men at the hotel take Dude hostage and Nathan Burdette offers to trade him later for Joe .
Stumpy opens fire at the jail , holding off Burdette 's men . Chance and Colorado then take Joe to make the trade for Dude at a warehouse , leaving Stumpy
behind because of his bad leg . During the trade , walking in opposite directions , Dude tackles Joe and they scuffle while a gunfight erupts . Stumpy turns up
and helps even the odds . Dude overpowers Joe . The lawmen throw dynamite sticks at the warehouse where Burdette and his men are based , shooting the sticks to
explode them . After a few explosions rock the warehouse , the criminals surrender . All is quiet in town as Chance gets reacquainted with Feathers , who models
a skimpy new costume she will be wearing in her new job singing at the hotel .
</item 1>

<item 2>
Rio Bravo (film)
( 1962 ) Man 's Favorite Sport ? ( 1964 ) Red Line 7000 ( 1965 ) El Dorado ( 1966 ) Rio Lobo ( 1970 ) VIAF : 316751659 GND : 4764446 - 1 SUDOC : 104486708 BNF :
cb145169957 ( data ) Retrieved from `` https://en.wikipedia.org/w/index.php?title=Rio_Bravo_(film)&oldid=817646117 '' Categories : 1959 films English - language
films 1950s Western ( genre ) films American Western ( genre ) films American buddy films Films scored by Dimitri Tiomkin Films about alcoholism Films directed
by Howard Hawks Films set in Texas Films shot in Arizona Films shot in California Screenplays by Leigh Brackett Warner Bros.
</item 2>

<item 3>
Rio Bravo (film)
Rio Bravo ( film ) - wikipedia Rio Bravo ( film ) Jump to : navigation , search For other uses , see Rio Bravo . Rio Bravo Theatrical release poster Directed by
Howard Hawks Produced by Howard Hawks Screenplay by Jules Furthman Leigh Brackett Based on `` Rio Bravo '' by B.H. McCampbell Starring John Wayne Dean Martin
Ricky Nelson Angie Dickinson Walter Brennan Ward Bond John Russell Music by Dimitri Tiomkin Cinematography Russell Harlan Edited by Folmar Blangsted Production
company Warner Bros . Distributed by Warner Bros . Release date March 18 , 1959 ( 1959 - 03 - 18 ) ( USA ) Running time 141 minutes Country United States
Language English Box office $5.2 million ( est . US / Canada rentals ) Rio Bravo is a 1959 American Western film produced and directed by Howard Hawks and
starring John Wayne , Dean Martin , Ricky Nelson , Angie Dickinson , Walter Brennan , and Ward Bond . Written by Jules Furthman and Leigh Brackett , based on
the short story `` Rio Bravo '' by B.H.
</item 3>

<item 4>
Rio Bravo (film)
Inspirations ( edit ) Feathers ' dialogue was occasionally inspired by the character of `` Slim '' ( Lauren Bacall ) in the 1944 To Have and Have Not , as when
, after the first kiss , she says : `` It 's better when two people do it , '' recalling the phrase `` It 's even better when you help ; '' and again later when
she says , `` I 'm hard to get -- you 're going to have to say you want me , '' recalling Slim 's `` I 'm hard to get , Steve -- all you have to do is ask me .
'' L'homme à l'étoile d'argent ( The Man with the Silver Star ) , a 1969 album from the French comics series Lt. Blueberry was directly inspired by Rio Bravo .
The plot is virtually the same . Blueberry plays the role of sheriff John T. Chance ; McClure , a whiskey - adoring old man , combines the roles of Dude and
Stumpy ; Dusty plays the role of Colorado ; Miss March , the teacher , plays the role of a less morally challenged Feathers ; and instead of the Burdettes , it
has the Bass brothers .
</item 4>

<item 5>
Rio Bravo (film)
The Nest , a 2002 film by Florent Emilio Siri , starring Samy Naceri , Benoît Magimel , Nadia Farès , Pascal Greggory , and Sami Bouajila , is a quasi-remake of
Assault on Precinct 13 . Borrachón : A Prequel to Rio Bravo , by Kevin Cullen , 2016 . This book is a prequel to the Rio Bravo story and film and provides the
complete background leading up to the classic tale . Music ( edit ) `` My Rifle , My Pony and Me '' -- sung by Dean Martin and Ricky Nelson `` Cindy '' -- sung
by Ricky Nelson , Dean Martin and Walter Brennan `` Rio Bravo '' -- sung by Dean Martin ( end credits ) Comic book adaption ( edit ) Dell Four Color # 1018 (
June 1959 ) See also ( edit ) List of American films of 1959 John Wayne filmography El Dorado ( 1966 ) Rio Lobo ( 1970 ) References ( edit ) Jump up ^ `` 1959 :
Probable Domestic Take '' , Variety , 6 January 1960 p. 34 .
</item 5>

<item 6>
Rio Bravo (film)
In Rio Bravo , Chance is surrounded by allies -- a deputy who is brave and good with a gun , despite recovering from alcoholism ( Dude ) , a young untried but
self - assured gunfighter ( Colorado ) , a limping `` crippled '' old man who is doggedly loyal ( Stumpy ) , a Mexican innkeeper ( Carlos ) , his wife (
Consuela ) , and an attractive young woman ( Feathers ) -- and repeatedly turns down aid from anyone he does not think is capable of helping him , though in the
final shootout they come to help him anyway . `` Who 'll turn up next ? '' Wayne asks amid the gunfire , to which Colorado replies : `` Maybe the girl with
another flower pot . '' Reception ( edit ) In the United Kingdom , Rio Bravo was not originally even reviewed for Sight & Sound ; in August 1976 , Leslie
Halliwell gave the film two out of four stars , describing it as a `` cheerfully overlong and slow - moving Western '' that was nevertheless `` very watchable
for those with time to spare '' .
</item 6>

<item 7>
Rio Bravo (film)
Rio Bravo is generally regarded as one of Hawks ' best , and is notable for its long opening scene which contains no dialogue . The film received favorable
reviews , and was successful , taking in over US $ 5.5 million . The musical score was composed by Dimitri Tiomkin . His score includes the hauntingly ominous
`` El Degüello '' theme , which is heard several times . The Colorado character identifies the tune as `` The Cutthroat Song '' . He relates that the song was
played on the orders of General Antonio López de Santa Anna to the Texans holed up in the Alamo , to signify that no quarter would be given to them . The tune
was used the following year in Wayne 's film , The Alamo . Composer Ennio Morricone recalled that director Sergio Leone asked him to write `` Dimitri Tiomkin
music '' for A Fistful of Dollars . The trumpet theme is similar to Tiomkin 's `` Degüello '' ( the Italian title of Rio Bravo was Un dollaro d'onore , A Dollar
of Honor ) .
</item 7>

<item 8>
Rio Bravo (film)
McCampbell , the film is about the sheriff of the town of Rio Bravo , Texas , who arrests the brother of a powerful local rancher to help his drunken deputy /
friend . With the help of a cripple and a young gunfighter , they hold off the rancher 's gang . Rio Bravo was filmed on location at Old Tucson Studios outside
Tucson , Arizona , in Technicolor . In 2014 , Rio Bravo was deemed `` culturally , historically , or aesthetically significant '' by the Library of Congress and
selected for preservation in the National Film Registry . Contents ( hide ) 1 Plot 2 Cast 3 Production 4 High Noon debate 5 Reception 6 Remakes and inspirations
6.1 Remakes 6.2 Inspirations 7 Music 8 Comic book adaption 9 See also 10 References 11 Further reading 12 External links Plot ( edit ) In the town of Rio Bravo
, Texas , sheriff 's deputy Dude ( Dean Martin ) , who has acquired the contemptuous nickname Borrachón ( pronounced ( bo.
</item 8>

<item 9>
Rio Bravo (film)
After a pair of sheer tights comes floating from a window to the street , a delighted Stumpy retrieves it , but Dude cautions him to mind his own business .
Cast ( edit ) John Wayne and Angie Dickinson in Rio Bravo John Wayne as John T. Chance Dean Martin as Dude Ricky Nelson as Colorado / Ryan Angie Dickinson as
Feathers Walter Brennan as Stumpy Ward Bond as Pat Wheeler John Russell as Nathan Burdette Pedro Gonzalez Gonzalez as Carlos Robante Estelita Rodriguez as
Consuela Robante Claude Akins as Joe Burdette Joseph Shimada as Burt Malcolm Atterbury and Harry Carey Jr. also receive a screen credit in the film 's opening ,
but their scenes were deleted from the final film . Production ( edit ) Ricky Nelson performing the song `` Get Along Home , Cindy '' in Rio Bravo Exteriors for
the film were shot at Old Tucson Studios , just outside Tucson . Filming took place in the summer of 1958 , and the movie 's credits gave 1958 as the year of
production , although the film was not released until 1959 .
</item 9>

<item 10>
Rio Bravo (film)
In this film , Robert Mitchum played the Dean Martin role , Arthur Hunnicutt the Walter Brennan character , and James Caan the Ricky Nelson role . Hawks again
named the Nelson / Caan character after a state ( in this case , Mississippi ) and in a wry , humorous twist on the original film , Hawks made him inept with
firearms , but skilled with a knife . The second remake , Rio Lobo , was made in 1970 with a plot much further off the original mold , starting with the absence
of a lawman - turned - drunkard character . This began with a Confederate train robbery of a Union gold shipment during the American Civil War , then moved to a
postwar Texas county thoroughly controlled by a rich , arrogant rancher . The heroes , with the exception of an old man similar to Brennan and Hunnicutt 's
characters in the previous pictures ( Jack Elam here ) , were complete outsiders . Along with Wayne and Elam , this movie starred Mexican film star Jorge Rivero
( as Frenchie ) , Christopher Mitchum ( Robert Mitchum 's son ) , and Jennifer O'Neill .
</item 10>
=== END ITEM id=fa639c256592cc0a ===

=== ITEM id=77479770d0623bf8 ===
Gold sentences and facts:
Sentence 0: Their first video arcade game was 1975 's EVR Race ; a handful of others followed in the next several years , Radar Scope and Donkey Kong being among the most famous of these . The early 1980s saw Nintendo 's video game division ( led by Yokoi ) creating some of its most famous arcade titles . The massively popular Donkey Kong was created in 1981 , with Shigeru Miyamoto as its mastermind , and released in the arcades and on the Atari 2600 , Intellivision , and ColecoVision video game systems ( although Nintendo themselves generally had no involvement with these early console ports ) . This release method would be used on several later Nintendo arcade games of this same period , including the original Mario Bros. ( not to be confused with the later Super Mario Bros . ) In addition to this arcade and dedicated console game activity , Nintendo was testing the consumer handheld video game waters with the Game & Watch .
  Fact 0.0: Nintendo's first video arcade game was EVR Race, released in 1975.
  Fact 0.1: A handful of other arcade games followed Nintendo's first video arcade game in the next several years.
  Fact 0.2: Radar Scope and Donkey Kong were among the most famous of those arcade games.
  Fact 0.3: In the early 1980s, Nintendo's video game division, led by Yokoi, created some of Nintendo's most famous arcade titles.
  Fact 0.4: Donkey Kong was massively popular.
  Fact 0.5: Donkey Kong was created in 1981.
  Fact 0.6: Shigeru Miyamoto was the mastermind behind Donkey Kong.
  Fact 0.7: Donkey Kong was released in the arcades.
  Fact 0.8: Donkey Kong was released on the Atari 2600, Intellivision, and ColecoVision video game systems.
  Fact 0.9: Nintendo themselves generally had no involvement with the early console ports of Donkey Kong.
  Fact 0.10: The release method used for Donkey Kong, arcade plus console, was used on several later Nintendo arcade games of the same period.
  Fact 0.11: The original Mario Bros. was one of these later Nintendo arcade games, and it is not to be confused with the later Super Mario Bros.
  Fact 0.12: Alongside its arcade and dedicated console game activity, Nintendo was testing the consumer handheld video game market with the Game & Watch.
Sentence 1: In 1972 , the first commercially available video game console , the Magnavox Odyssey , had a light gun accessory , the Shooting Gallery . This was the first involvement of Nintendo in video games . According to Martin Picard in the International Journal of Computer Game Research : `` in 1971 , Nintendo had -- even before the marketing of the first home console in the United States -- an alliance with the American pioneer Magnavox to develop and produce optoelectronic guns for the Odyssey ( released in 1972 ) , since it was similar to what Nintendo was able to offer in the Japanese toy market in 1970s '' .
  Fact 1.0: The Magnavox Odyssey was the first commercially available video game console, and it was released in 1972.
  Fact 1.1: In 1972, the Magnavox Odyssey had a light gun accessory called the Shooting Gallery.
  Fact 1.2: The Shooting Gallery was the first involvement of Nintendo in video games.
  Fact 1.3: According to Martin Picard in the International Journal of Computer Game Research, in 1971 Nintendo had an alliance with the American pioneer Magnavox.
  Fact 1.4: The alliance was formed even before the marketing of the first home console in the United States.
  Fact 1.5: The alliance was to develop and produce optoelectronic guns for the Odyssey.
  Fact 1.6: The Odyssey was released in 1972.
  Fact 1.7: The alliance existed because Nintendo was able to offer something similar in the Japanese toy market in the 1970s.

Retrieved items:
<item 1>
History of Nintendo
History of Nintendo - wikipedia History of Nintendo Jump to : navigation , search This article needs additional citations for verification . Please help improve
this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( November 2006 ) ( Learn how and when to remove this
template message ) The history of Nintendo traces back to 1889 , when it was founded to produce handmade hanafuda cards . Nintendo Co. , Ltd. ( 任天堂 株式 会社 ,
Nintendō kabushikigaisha ) is a Japanese multinational consumer electronics company headquartered in Kyoto , Japan . It eventually became one of the most
prominent figures in today 's video game industry , being the world 's largest video game company by revenue .
</item 1>

<item 2>
History of Nintendo
Nintendo now began to experiment in other areas of business using the newly injected capital . During the period of time between 1963 and 1968 , Nintendo set up
a taxi company , `` love hotel '' chain , food company ( trying to sell instant rice , similar to instant noodles ) and several other things ( including a
vacuum cleaner , Chiritory , which later appeared as a two - player game in WarioWare , Inc. : Mega Microgame $ in 2003 ) . All these ventures eventually failed
, except toymaking , where they had some earlier experience from selling playing cards . In 1964 , while Japan was experiencing an economic boom due to the
Tokyo Olympics , the playing card business reached its saturation point . Japanese households stopped buying playing cards , and the price of Nintendo stock
fell from 900 yen to 60 yen . In 1965 , Nintendo hired Gunpei Yokoi as a maintenance engineer for the assembly line . However , Yokoi soon became famous for
much more than his ability to repair conveyor belts .
</item 2>

<item 3>
History of Nintendo
Contents ( hide ) 1 Origin 2 1889 -- 1965 : As a card company 3 1966 -- 1972 : Toy company and new ventures 4 1972 -- present : Electronic games 4.1 Color TV -
Game 6 4.2 Game & Watch and Nintendo Entertainment System 4.3 Game Boy and Super NES 4.4 Nintendo 64 and Game Boy Color 4.5 Game Boy Advance and GameCube 4.6
Nintendo DS and Wii 4.7 Nintendo 3DS , Wii U , and Nintendo Switch 5 References 6 Further reading 7 External links Origin ( edit ) Nintendo poster from late
Meiji Era Nintendo started out as a small Japanese company , founded by Fusajiro Yamauchi on September 23 , 1889 , as Nintendo Koppai . Based in Kyoto , Japan ,
the business produced and marketed Hanafuda cards . The name `` Nintendo '' is commonly assumed to mean `` leave luck to heaven '' , but there are no historical
records to validate this assumption .
</item 3>

<item 4>
History of Nintendo
The first issue published was the July / August edition , which spotlighted the NES game Super Mario Bros. 2 . Nintendo Power has since ceased publication with
its December 2012 edition . Game Boy and Super NES ( edit ) Main articles : Game Boy and Super Nintendo Entertainment System The original Game Boy In 1989 ,
Nintendo ( which had seen a large amount of success from the Game & Watch ) released the Game Boy ( both created by Gunpei Yokoi ) , along with the accompanying
game Tetris . Due to the price , the game and its durability ( unlike the prior Microvision from Milton Bradley Company , which was prone to static and screen
rot ) , the Game Boy line sold extremely well , eventually amassing sales of 118 million units . Later , Super Mario Land was also released for the Game Boy ,
which sold 14 million copies worldwide . 1989 was also the year that Nintendo announced a sequel to the Famicom , to be called the Super Famicom . The last
major first - party game for the NES , Super Mario Bros. 3 , was released in early 1990 , and went on to sell over 18 million units .
</item 4>

<item 5>
History of Nintendo
In 1982 , Nintendo developed a prototype system dubbed the `` Advanced Video System '' ( or AVS for short ) and had controllers much like the NES . There were
accessories such as a tape drive , a joystick and a lightgun , and along with all of that , the system was made of a computer , much like the Atari 400 ,
Commodore Vic - 20 and Commodore 64 . It was never released and is on display at the Nintendo World Store in New York . In July 1983 , Nintendo released the
Famicom ( Family Computer ) system in Japan , which was its first attempt at a cartridge - based video game console . The system sold over 500,000 units within
two months at a price around $100 USD . However , after a few months of favorable sales , Nintendo received complaints that some Famicom consoles would freeze
when the player attempted to play certain games . The fault was found in a malfunctioning chip and Nintendo decided to recall all Famicom units that were
currently on store shelves , which ultimately cost them approximately half a million USD .
</item 5>

<item 6>
History of Nintendo
Konami , the first third - party company that was allowed to make cartridges for the Famicom , would later circumvent this rule by creating a spinoff company ,
Ultra Games , to release additional games in a single year . Other manufacturers soon employed the same tactic . Also in 1985 , Super Mario Bros. was released
for the Famicom in Japan and became a large success . Nintendo test marketed the Nintendo Entertainment System in the New York area on October 18 , 1985 . They
expanded the test to Los Angeles in February 1986 , followed by tests in Chicago and San Francisco . They would go national by the end of 1986 , along with 15
games , sold separately . In the US and Canada , it outsold its competitors by a wide margin . This was also the year that Metroid and Super Mario Bros. 2 ( the
Japanese version ) were released . In 1987 , The Legend of Zelda was released to much critical acclaim . In 1988 , Nintendo of America unveiled Nintendo Power ,
a monthly news and strategy magazine from Nintendo that served to advertise new games .
</item 6>

<item 7>
History of Nintendo
This was the first involvement of Nintendo in video games . According to Martin Picard in the International Journal of Computer Game Research : `` in 1971 ,
Nintendo had -- even before the marketing of the first home console in the United States -- an alliance with the American pioneer Magnavox to develop and
produce optoelectronic guns for the Odyssey ( released in 1972 ) , since it was similar to what Nintendo was able to offer in the Japanese toy market in 1970s
'' . In 1973 , its focus shifted to family entertainment venues with the Laser Clay Shooting System , using the same light gun technology used in Nintendo 's
Kousenjuu series of toys , and set up in abandoned bowling alleys . Following some success , Nintendo developed several more light gun machines for the emerging
arcade scene . While the Laser Clay Shooting System ranges had to be shut down following excessive costs , Nintendo had found a new market . In 1974 , Nintendo
secured the rights to distribute the Magnavox Odyssey video game console in Japan .
</item 7>

<item 8>
History of Nintendo
These devices had a much higher novelty value than traditional toys , allowing Nintendo to charge a higher price margin for each product . Yokoi went on to
develop many other toys , including the Ten Billion Barrel puzzle , a baseball throwing machine called the Ultra Machine , and a Love Tester . Another invention
of his , in collaboration with Masayuki Uemura from Sharp , was the Nintendo Beam Gun Game , the precursor to the NES Zapper ( which was used in games like Duck
Hunt . ) . Nintendo released the first solar - powered light gun , the Nintendo Beam Gun , in 1970 ; this was the first commercially available light - gun for
home use , produced in partnership with Sharp . In 1972 , Nintendo released the Ele - Conga , one of the first programmable drum machines . It could play
pre-programmed rhythms from disc - shaped punch cards , which could be altered or programmed by the user , to play different patterns . 1972 -- present :
electronic games ( edit ) In 1972 , the first commercially available video game console , the Magnavox Odyssey , had a light gun accessory , the Shooting
Gallery . This was the first involvement of Nintendo in video games .
</item 8>

<item 9>
History of Nintendo
Color TV - game 6 ( edit ) Main article : Color TV Game Nintendo 's first video game system , the Color TV Game 6 Nintendo at this time saw how successful video
games were and began to dabble in them . Their first step in that field was to secure the rights to distribute the Magnavox Odyssey in Japan in 1975 . At the
time , home video game consoles were extremely rare -- even the seminal Atari Pong console had yet to be produced . After experiencing reasonable success at
this , Nintendo began developing its own video games , both for home and for arcades . In the 1970s , Mitsubishi Electric proposed joint development of `` Color
TV Game Machine '' . In 1977 , they released `` Color TV Game 6 '' and `` Color TV Game 15 '' ( 6 and 15 indicates the number of games ) . Their first video
arcade game was 1975 's EVR Race ; a handful of others followed in the next several years , Radar Scope and Donkey Kong being among the most famous of these .
The early 1980s saw Nintendo 's video game division ( led by Yokoi ) creating some of its most famous arcade titles .
</item 9>

<item 10>
History of Nintendo
The massively popular Donkey Kong was created in 1981 , with Shigeru Miyamoto as its mastermind , and released in the arcades and on the Atari 2600 ,
Intellivision , and ColecoVision video game systems ( although Nintendo themselves generally had no involvement with these early console ports ) . This release
method would be used on several later Nintendo arcade games of this same period , including the original Mario Bros. ( not to be confused with the later Super
Mario Bros . ) In addition to this arcade and dedicated console game activity , Nintendo was testing the consumer handheld video game waters with the Game &
Watch . Game & Watch and Nintendo Entertainment system ( edit ) Main articles : Game & Watch , Nintendo Entertainment System , and History of the Nintendo
Entertainment System Game & Watch or G&W is a line of handheld electronic games produced by Nintendo from 1980 to 1991 . Created by game designer Gunpei Yokoi ,
each Game & Watch features a single game to be played on an LCD screen in addition to a clock and / or an alarm . It was the earliest Nintendo product to garner
major success , with 43.4 million units sold worldwide .
</item 10>
=== END ITEM id=77479770d0623bf8 ===

=== ITEM id=4d51297ab5dc1e61 ===
Gold sentences and facts:
Sentence 0: Roger Federer was the defending champion and successfully defended his title , defeating Marin Čilić in the final , 6 -- 2 , 6 -- 7 , 6 -- 3 , 3 -- 6 , 6 -- 1 . It was Federer 's 20th Grand Slam singles title and record - equalling sixth Australian Open men 's singles title ( tied with Roy Emerson and Novak Djokovic ) . With the win , Federer became the first male player to win at least six titles at two Grand Slam tournaments ( six at the Australian Open and eight at Wimbledon ) . Federer became the oldest man to win a Grand Slam singles title in the Open era since Ken Rosewall in 1972 . The time span between Federer 's first Grand slam glory at Wimbledon and this latest , almost 15 years , is an Open era record in the men 's singles field . This was also the 10th time that Federer has defended a Grand Slam title , with the previous time being at the 2008 US Open . Čilić became the first Croatian player to reach a singles final at the Australian Open .
  Fact 0.0: Roger Federer was the defending champion.
  Fact 0.1: Roger Federer successfully defended his title.
  Fact 0.2: Roger Federer defeated Marin Čilić in the final.
  Fact 0.3: The final score was 6-2, 6-7, 6-3, 3-6, 6-1.
  Fact 0.4: The win was Roger Federer's 20th Grand Slam singles title.
  Fact 0.5: The win was Roger Federer's record-equalling sixth Australian Open men's singles title.
  Fact 0.6: Roger Federer's sixth Australian Open men's singles title tied him with Roy Emerson and Novak Djokovic.
  Fact 0.7: With the win, Roger Federer became the first male player to win at least six titles at two Grand Slam tournaments.
  Fact 0.8: Roger Federer won six of those titles at the Australian Open.
  Fact 0.9: Roger Federer won eight of those titles at Wimbledon.
  Fact 0.10: Roger Federer became the oldest man to win a Grand Slam singles title in the Open era since Ken Rosewall won one in 1972.
  Fact 0.11: The time span between Roger Federer's first Grand Slam title at Wimbledon and his latest Grand Slam title is almost 15 years.
  Fact 0.12: That time span is an Open era record in men's singles.
  Fact 0.13: This was Roger Federer's 10th defense of a Grand Slam title.
  Fact 0.14: Roger Federer's previous Grand Slam title defense was at the 2008 US Open.
  Fact 0.15: Marin Čilić became the first Croatian player to reach a singles final at the Australian Open.
Sentence 1: Men 's Singles 2018 Australian Open Champion Roger Federer Runner - up Marin Čilić Final score 6 -- 2 , 6 -- 7 , 6 -- 3 , 3 -- 6 , 6 -- 1 Details Draw 128 ( 16 Q / 8 WC ) Seeds 32 Events Singles men women boys girls Doubles men women mixed boys girls Legends men women mixed WC Singles men women quad WC Doubles men women quad ← 2017 Australian Open 2019 →
  Fact 1.0: The men's singles champion of the 2018 Australian Open was Roger Federer.
  Fact 1.1: The men's singles runner-up of the 2018 Australian Open was Marin Čilić.
  Fact 1.2: The 2018 Australian Open men's singles final score was 6-2, 6-7, 6-3, 3-6, 6-1.
  Fact 1.3: The 2018 Australian Open men's singles draw had 128 entries (16 qualifiers and 8 wild cards).
  Fact 1.4: The 2018 Australian Open men's singles had 32 seeds.
  Fact 1.5: The previous edition was the 2017 Australian Open.
  Fact 1.6: The next edition was the 2019 Australian Open.

Retrieved items:
<item 1>
2018 Australian Open – Men's Singles
2003 ) Andre Agassi ( 2004 ) Roger Federer ( 2005 ) Marat Safin ( 2006 ) Roger Federer ( 2007 ) Roger Federer ( 2008 ) Novak Djokovic ( 2009 ) Rafael Nadal (
2010 ) Roger Federer ( 2011 ) Novak Djokovic ( 2012 ) Novak Djokovic ( 2013 ) Novak Djokovic ( 2014 ) Stanislas Wawrinka ( 2015 ) Novak Djokovic ( 2016 ) Novak
Djokovic ( 2017 ) Roger Federer ( 2018 ) Roger Federer 2018 ATP World Tour `` 2017 2019 '' Grand Slam events Australian Open ( S , D , X ) French Open ( S , D ,
X ) Wimbledon ( S , D , X ) US Open ( S , D , X ) ATP World Tour Masters 1000 Indian Wells ( S , D ) Miami ( S , D ) Monte Carlo ( S , D ) Madrid ( S , D ) Rome
( S , D ) Montreal ( S , D ) Cincinnati ( S ,
</item 1>

<item 2>
2018 Australian Open – Men's Singles
With the win , Federer became the first male player to win at least six titles at two Grand Slam tournaments ( six at the Australian Open and eight at Wimbledon
) . Federer became the oldest man to win a Grand Slam singles title in the Open era since Ken Rosewall in 1972 . The time span between Federer 's first Grand
slam glory at Wimbledon and this latest , almost 15 years , is an Open era record in the men 's singles field . This was also the 10th time that Federer has
defended a Grand Slam title , with the previous time being at the 2008 US Open . Čilić became the first Croatian player to reach a singles final at the
Australian Open . Federer and Rafael Nadal were in contention for the ATP no . 1 singles ranking at the start of the tournament . Nadal retained the top ranking
by reaching the quarterfinals .
</item 2>

<item 3>
2018 Australian Open – Men's Singles
D ) Rome ( S , D ) Montreal ( S , D ) Cincinnati ( S , D ) Shanghai ( S , D ) Paris ( S , D ) ATP World Tour 500 series Rotterdam ( S , D ) Rio de Janeiro ( S ,
D ) Dubai ( S , D ) Acapulco ( S , D ) Barcelona ( S , D ) London ( S , D ) Halle ( S , D ) Hamburg ( S , D ) Washington D.C. ( S , D ) Beijing ( S , D ) Tokyo
( S , D ) Vienna ( S , D ) Basel ( S , D ) ATP World Tour 250 series Brisbane ( S , D ) Doha ( S , D ) Pune ( S , D ) Auckland ( S , D ) Sydney ( S , D )
Montpellier ( S , D ) Sofia ( S , D ) Quito ( S , D ) Buenos Aires ( S , D ) Uniondale ( S , D ) Delray Beach ( S , D ) Marseille ( S , D ) São Paulo ( S , D )
Houston ( S , D ) Marrakesh ( S , D ) Budapest ( S , D ) Estoril ( S ,
</item 3>

<item 4>
2018 Australian Open – Men's Singles
) ) Roscoe Tanner ( 1977 ( Dec ) ) Vitas Gerulaitis ( 1978 ) Guillermo Vilas ( 1979 ) Guillermo Vilas ( 1980 ) Brian Teacher ( 1981 ) Johan Kriek ( 1982 ) Johan
Kriek ( 1983 ) Mats Wilander ( 1984 ) Mats Wilander ( 1985 ) Stefan Edberg ( 1986 ) Not Held ( 1987 ) Stefan Edberg ( 1988 ) Mats Wilander ( 1989 ) Ivan Lendl (
1990 ) Ivan Lendl ( 1991 ) Boris Becker ( 1992 ) Jim Courier ( 1993 ) Jim Courier ( 1994 ) Pete Sampras ( 1995 ) Andre Agassi ( 1996 ) Boris Becker ( 1997 )
Pete Sampras ( 1998 ) Petr Korda ( 1999 ) Yevgeny Kafelnikov ( 2000 ) Andre Agassi ( 2001 ) Andre Agassi ( 2002 ) Thomas Johansson ( 2003 ) Andre Agassi ( 2004
) Roger Federer ( 2005 )
</item 4>

<item 5>
2018 Australian Open – Men's Singles
2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 Australian Open men 's singles drawsheets Pre-Open era 1905 1906 1907 1908 1909 1910 1911 1912 1913
1914 1915 1916 -- 1918 ( WWI ) 1919 1920 1921 1922 1923 1924 1925 1926 1927 1928 1929 1930 1931 1932 1933 1934 1935 1936 1937 1938 1939 1940 1941 -- 1945 ( WWII
) 1946 1947 1948 1949 1950 1951 1952 1953 1954 1955 1956 1957 1958 1959 1960 1961 1962 1963 1964 1965 1966 1967 1968 Open era 1969 1970
</item 5>

<item 6>
2018 Australian Open – Men's Singles
29 January 2018 . Retrieved 7 February 2018 . Jump up ^ `` Roger Federer beats Marin Cilic to win Australian Open 2018 men 's singles final -- as it happened ''
. Guardian . 28 January 2018 . Retrieved 7 February 2018 . Jump up ^ `` From Milan To Melbourne , Chung Keeps Surging '' . atpworldtour.com. 24 January 2018 .
Retrieved 24 January 2018 . Jump up ^ Ryan , Peter ( 2018 - 01 - 23 ) . `` Hyeon Chung takes South Korean tennis to the world '' . The Sydney Morning Herald .
Retrieved 2018 - 01 - 24 . Jump up ^ `` Five takeaways from Hyeon Chung 's Aussie Open win over Novak Djokovic '' . Tennis.com. 22 January 2018 . Retrieved 23
January 2018 . Australian Open ( Open Era ) singles drawsheets Men 's singles 1969 1970 1971 1972 1973 1976 1977 Jan
</item 6>

<item 7>
2018 Australian Open – Men's Singles
2018 Australian Open -- men 's singles - wikipedia 2018 Australian Open -- men 's singles Jump to : navigation , search Men 's Singles 2018 Australian Open
Champion Roger Federer Runner - up Marin Čilić Final score 6 -- 2 , 6 -- 7 , 6 -- 3 , 3 -- 6 , 6 -- 1 Details Draw 128 ( 16 Q / 8 WC ) Seeds 32 Events Singles
men women boys girls Doubles men women mixed boys girls Legends men women mixed WC Singles men women quad WC Doubles men women quad ← 2017 Australian Open 2019
→ Main article : 2018 Australian Open Roger Federer was the defending champion and successfully defended his title , defeating Marin Čilić in the final , 6 -- 2
, 6 -- 7 , 6 -- 3 , 3 -- 6 , 6 -- 1 . It was Federer 's 20th Grand Slam singles title and record - equalling sixth Australian Open men 's singles title ( tied
with Roy Emerson and Novak Djokovic ) .
</item 7>

<item 8>
2018 Australian Open – Men's Singles
1965 1966 1967 1968 Open era 1969 1970 1971 1972 1973 1976 1977 1977 1978 1979 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998
1999 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 Australian Open men 's singles champions ( 1969 ) Rod Laver ( 1970 )
Arthur Ashe ( 1971 ) Ken Rosewall ( 1972 ) Ken Rosewall ( 1973 ) John Newcombe ( 1974 ) Jimmy Connors ( 1975 ) John Newcombe ( 1976 ) Mark Edmondson ( 1977 (
Jan ) ) Roscoe Tanner ( 1977 ( Dec ) ) Vitas Gerulaitis (
</item 8>

<item 9>
2018 Australian Open – Men's Singles
2 Section 6 4.4. 3 Section 7 4.4. 4 Section 8 5 References Seeds ( edit ) 01 . Rafael Nadal ( Quarterfinals , retired ) 02 . Roger Federer ( Champion ) 03 .
Grigor Dimitrov ( Quarterfinals ) 04 . Alexander Zverev ( Third round ) 05 . Dominic Thiem ( Fourth round ) 06 . Marin Čilić ( Final ) 07 . David Goffin (
Second round ) 08 . Jack Sock ( First round ) 09 . Stan Wawrinka ( Second round ) 10 . Pablo Carreño Busta ( Fourth round ) 11 . Kevin Anderson ( First round )
12 . Juan Martín del Potro ( Third round ) 13 . Sam Querrey ( Second round ) 14 . Novak Djokovic ( Fourth round ) 15 . Jo - Wilfried Tsonga ( Third round ) 16 .
John Isner ( First round ) 17 .
</item 9>

<item 10>
2018 Australian Open – Men's Singles
1970 1971 1972 1973 1976 1977 Jan Dec 1978 1979 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005
2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 Women 's singles 1969 1970 1971 1972 1973 1976 1977 Jan Dec 1978 1979 1980 1981 1982 1984 1985 1986
1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012
</item 10>
=== END ITEM id=4d51297ab5dc1e61 ===

=== ITEM id=75d9ecfb6dda7a3d ===
Gold sentences and facts:
Sentence 0: Teri Garr as Mrs. Ramsay , Alec 's mother
  Fact 0.0: Teri Garr played the role of Mrs. Ramsay
  Fact 0.1: Mrs. Ramsay is Alec's mother

Retrieved items:
<item 1>
The Black Stallion (film)
In 2002 , it was selected for preservation in the United States National Film Registry by the Library of Congress as being `` culturally , historically , or
aesthetically significant '' . Also , the film is recognized by American Film Institute in these lists : 2001 : AFI 's 100 Years ... 100 Thrills -- Nominated
2005 : AFI 's 100 Years of Film Scores -- Nominated 2006 : AFI 's 100 Years ... 100 Cheers -- # 64 Sequel , prequel and television series ( edit ) The film was
followed in 1983 by a sequel , The Black Stallion Returns , which also starred Reno . There was also a television series called The Adventures of the Black
Stallion which aired from 1990 to 1993 and starred Mickey Rooney and Richard Ian Cox . In 2003 , a 50 - minute prequel called The Young Black Stallion , was
shot and released for IMAX theaters . See also ( edit ) The Black Stallion books References ( edit ) Jump up ^ `` The Black Stallion , Box Office Information ''
. The Numbers .
</item 1>

<item 2>
The Black Stallion (film)
Alec 's father rescues Alec from his attacker but the life jacket is lost and father & son are separated . The black stallion breaks free from his compartment
and then jumps into the sea . Alec himself is swept overboard by a gigantic wave . Once in the water , he swims toward the stallion and managed to grab hold of
the ropes that were previously restraining the stallion . Alec wakes on the shore of a deserted island and starts to explore . He finds the stallion caught in
his restraints with the ropes stuck between the rocks . With his knife , Alec manages to free the stallion once again and the stallion runs away . For a time ,
the two keep their distance . Alec discovers means to survive by catching fish and seaweed . As Alec suddenly faces a cobra eye to eye , the Black comes to the
rescue and kills the snake , only to run off again . By now , Alec decides to try to get closer to the horse and offer him some seaweed . The hungry stallion
finds himself unable to resist , but visibly struggles with his distrust for humans .
</item 2>

<item 3>
The Black Stallion (film)
The Black stallion ( film ) - wikipedia The Black stallion ( film ) Jump to : navigation , search The Black Stallion 1979 poster Directed by Carroll Ballard
Produced by Fred Roos Francis Ford Coppola Tom Sternberg Screenplay by Melissa Mathison Jeanne Rosenberg William D. Wittliff Based on The Black Stallion by
Walter Farley Starring Kelly Reno Mickey Rooney Teri Garr Music by Carmine Coppola Cinematography Caleb Deschanel Edited by Robert Dalva Production company
American Zoetrope Distributed by United Artists Release date October 17 , 1979 ( 1979 - 10 - 17 ) Running time 118 minutes Country United States Language
English Budget $2.7 million Box office $37.8 million The Black Stallion is a 1979 American adventure film based on the 1941 classic children 's novel The Black
Stallion by Walter Farley . But the film starts in 1946 , five years after the book was published . It tells the story of Alec Ramsey , who is shipwrecked on a
deserted island with a wild Arabian stallion whom he befriends .
</item 3>

<item 4>
The Black Stallion (film)
Henry allows Alec to keep the Black in his barn . When Alec wonders how fast the Black is , Alec and Henry decide to train the Black for the racetrack , while
Henry teaches Alec how to be a jockey . The Black surprises Henry with his speed . Henry immediately starts plotting to get the Black into a match race between
two of the country 's current champions , to be held at Santa Anita , California . To do that , he sets up a secret demonstration at night where the press can
witness his speed , keeping the identity of Alec and the Black secret . The news about the mystery horse is soon widespread and the Black is entered into the
race . The race is the most anticipated horse - racing event of the year . Before the two champions and the Black enter the starting gate , the Black gets into
a fight with one of his opponents , wounding his leg . Alec does not see the wound until he is in the gate . As he dismounts , the bell rings and the horses
take off . Alec desperately tries to stay on his horse and trying to stop him . He falls behind , but the Black wo n't stop .
</item 4>

<item 5>
The Black Stallion (film)
After being rescued , they are set on entering a race challenging two champion horses . The film is adapted by Melissa Mathison , Jeanne Rosenberg and William
D. Wittliff . It is directed by Carroll Ballard . The movie stars Kelly Reno , Mickey Rooney , Teri Garr , Hoyt Axton , and the Arabian horse Cass Ole . The
film features music by Carmine Coppola , the father of Hollywood producer Francis Ford Coppola , who was the executive producer of the film . Contents ( hide )
1 Plot 2 Cast 2.1 Horses 3 Awards and honors 3.1 Academy Awards 3.2 Golden Globe Awards 3.3 British Academy Awards 3.4 LA Film Critics Awards 3.5 Others 4
Sequel , prequel and television series 5 See also 6 References 7 External links Plot ( edit ) In 1946 Alec Ramsey is aboard the steamer Drake off the coast of
North Africa , where he sees a wild black stallion being forced into a makeshift stable compartment and heavily restrained by ropes leading to his halter .
</item 5>

<item 6>
The Black Stallion (film)
He falls behind , but the Black wo n't stop . When Alec regains his balance , the Black is well on his way to catch up with his opponents . Alec now encourages
the Black to run as fast as he can , remembering the wild rides on the island , as they catch up . The Black eventually wins by two lengths . Cast ( edit )
Kelly Reno as Alec Ramsay Mickey Rooney as Henry Dailey Teri Garr as Mrs. Ramsay , Alec 's mother Hoyt Axton as Mr. Ramsay , Alec 's father Clarence Muse as
Snoe Michael Higgins as Jim Neville Ed McNamara as Jake Doghmi Larbi as Arab John Burton as Jockey No. 1 John Buchanan as Jockey No.
</item 6>

<item 7>
The Black Stallion (film)
American Film Institute . Retrieved 2016 - 08 - 20 . External links ( edit ) Official website The Black Stallion on IMDb The Black Stallion at AllMovie The
Black Stallion at Rotten Tomatoes The Black Stallion at Box Office Mojo The Black Stallion at the TCM Movie Database The Black Stallion by Walter Farley Books
The Black Stallion book series Films The Black Stallion ( 1979 ) The Black Stallion Returns ( 1983 ) The Young Black Stallion ( 2003 ) Television The Adventures
of the Black Stallion ( 1990 - 1993 ) Related Cass Ole El Mokhtar Docs Keepin Time Films directed by Carroll Ballard The Black Stallion ( 1979 ) Never Cry Wolf
( 1983 ) Nutcracker : The Motion Picture ( 1986 ) Wind ( 1992 ) Fly Away Home ( 1996 ) Duma ( 2005 ) Retrieved from `` https://en.wikipedia.org/w/index.
</item 7>

<item 8>
The Black Stallion (film)
1 John Buchanan as Jockey No. 2 Kristen Vigard as Becky Fausto Tozzi as Rescue Captain John Karlsen as Archeologist Leopoldo Trieste as Priest Frank Cousins as
African Chieftain Don Hudson as Zaurog Marne Maitland as Drake Captain Tom Dahlgren as Veterinarian Horses ( edit ) Cass Ole , a champion Arabian stallion , was
featured in most of the movie 's scenes , with Fae Jur , another black Arabian stallion , being his main double . Fae Jur 's main scene is the one where Alec is
trying to gain the trust of the Black on the beach . Two other stunt doubles were used for running , fighting and swimming scenes . El Mokhtar , an Egyptian
Arabian racehorse , was the producers ' first choice to portray the Black , but they were unable to secure his services for the film from his owners , who
declined any offers . He does appear in The Black Stallion Returns , alongside Cass Ole , by which time the studio bought out the syndicate of owners in order
to secure El Mokhtar 's services .
</item 8>

<item 9>
The Black Stallion (film)
wikipedia.org/w/index.php?title=The_Black_Stallion_(film)&oldid=836268326 '' Categories : 1979 films English - language films American children 's films Films
about horses United States National Film Registry films United Artists films American Zoetrope films Directorial debut films Films shot in Sardinia Films
directed by Carroll Ballard Films that won the Best Sound Editing Academy Award Horse racing films Screenplays by Melissa Mathison American films Films set on
islands Films set in 1946 Hidden categories : Pages using div col with deprecated parameters Talk Contents About Wikipedia Català Dansk Deutsch Español فارسی
Français Bahasa Indonesia Italiano Bahasa Melayu Nederlands 日本 語 Polski Português Русский Svenska 6 more Edit links This page was last edited on 13 April 2018 ,
at 18 : 25 . About Wikipedia
</item 9>

<item 10>
The Black Stallion (film)
The hungry stallion finds himself unable to resist , but visibly struggles with his distrust for humans . Eventually , the hunger wins and he takes Alec 's
offer ; their bond has been sealed and the two are now inseparable . Alec even manages to ride the unbroken horse , after many times falling off the horse . One
day , a fishing ship arrives , rescuing both Alec and the stallion . Back home in Flushing , NY , Alec is given a hero 's welcome . The Black has a temporary
home in Alec 's backyard , but a garbage man not knowing that there is a wild horse in the backyard is chased by the Black , who races off down the street after
being spooked by a passing car . Alec chases after him through every part of town , but loses track of him . The next day , Alec meets Snoe ( and Napoleon ) who
tell him where the Black is . Alec finds the stallion in the barn of Henry Dailey , a retired jockey and racehorse trainer , who apparently spent all night
catching the Black . Henry allows Alec to keep the Black in his barn .
</item 10>
=== END ITEM id=75d9ecfb6dda7a3d ===

=== ITEM id=0ff65f84339efb6b ===
Gold sentences and facts:
Sentence 0: Character Actor Job House Song ( s ) Miss Hoolie Julie Wilson Nimmo Nursery teacher Green house Nursery Song : Everybody everyone Day off song : What do you want to do today ( Series 1 -- 2 ) Strike up the band ( Series 3 -- 4 ) Archie the Inventor Miles Jupp Inventor Pink castle I 'm Archie the Inventor ( Series 1 - 4 ) Great Inventions , Groovy Solutions ( Series 3 -- 4 ) Josie Jump Buki Akib ( series 1 - 2 ) Kasia Haddad ( Series 3 -- 4 ) Fitness instructor Yellow house Josie Jump is My Name ( Jump a Little Higher ) ( Series 1 - 4 ) Cheer You Up ( Series 3 -- 4 ) Edie McCredie Juliet Cadzow Bus driver Blue garage / house When I Honk My Horn ( Series 1 - 4 ) Let Me Take You on a Journey ( Series 3 -- 4 ) PC Plum Andrew Agnew Policeman Black and white police station I 'm PC Plum ( Series 1 -- 4 ) Follow the Clue ( Series 3 -- 4 ) Spencer the Painter Rodd Christensen Painter / Musician Orange house Climbing Up My Musical Ladder ( Series 1 -- 4 ) If You Need a Little Rhythm ( Series 3 -- 4 ) Suzie Sweet Mary Riggans Shop and café workers Red shop I 'm Suzie Sweet , I 'm Penny Pocket ( Series 1 -- 4 ) Suzie 's Cooking ( Series 3 -- 4 ) Penny Pocket Kim Tserkezie I 'm Suzie Sweet , I 'm Penny Pocket ( Series 1 -- 4 ) Sort It ( Series 3 -- 4 )
  Fact 0.0: Miss Hoolie is played by Julie Wilson Nimmo.
  Fact 0.1: Miss Hoolie's job is nursery teacher.
  Fact 0.2: Miss Hoolie's house is the Green house (Nursery).
  Fact 0.3: Miss Hoolie's Nursery Song is Everybody everyone.
  Fact 0.4: Miss Hoolie's Day off song is What do you want to do today (Series 1-2).
  Fact 0.5: Miss Hoolie's Day off song is Strike up the band (Series 3-4).
  Fact 0.6: Archie the Inventor is played by Miles Jupp.
  Fact 0.7: Archie the Inventor's job is inventor.
  Fact 0.8: Archie the Inventor's house is the Pink castle.
  Fact 0.9: Archie the Inventor's song is I'm Archie the Inventor (Series 1-4).
  Fact 0.10: Archie the Inventor's song is Great Inventions, Groovy Solutions (Series 3-4).
  Fact 0.11: Josie Jump is played by Buki Akib in Series 1-2.
  Fact 0.12: Josie Jump is played by Kasia Haddad in Series 3-4.
  Fact 0.13: Josie Jump's job is fitness instructor.
  Fact 0.14: Josie Jump's house is the Yellow house.
  Fact 0.15: Josie Jump's song is Josie Jump is My Name (Jump a Little Higher) (Series 1-4).
  Fact 0.16: Josie Jump's song is Cheer You Up (Series 3-4).
  Fact 0.17: Edie McCredie is played by Juliet Cadzow.
  Fact 0.18: Edie McCredie's job is bus driver.
  Fact 0.19: Edie McCredie's house is the Blue garage / house.
  Fact 0.20: Edie McCredie's song is When I Honk My Horn (Series 1-4).
  Fact 0.21: Edie McCredie's song is Let Me Take You on a Journey (Series 3-4).
  Fact 0.22: PC Plum is played by Andrew Agnew.
  Fact 0.23: PC Plum's job is policeman.
  Fact 0.24: PC Plum's house is the Black and white police station.
  Fact 0.25: PC Plum's song is I'm PC Plum (Series 1-4).
  Fact 0.26: PC Plum's song is Follow the Clue (Series 3-4).
  Fact 0.27: Spencer the Painter is played by Rodd Christensen.
  Fact 0.28: Spencer the Painter's job is painter and musician.
  Fact 0.29: Spencer the Painter's house is the Orange house.
  Fact 0.30: Spencer the Painter's song is Climbing Up My Musical Ladder (Series 1-4).
  Fact 0.31: Spencer the Painter's song is If You Need a Little Rhythm (Series 3-4).
  Fact 0.32: Suzie Sweet is played by Mary Riggans.
  Fact 0.33: Suzie Sweet's job is shop and café worker.
  Fact 0.34: Suzie Sweet's house is the Red shop.
  Fact 0.35: Suzie Sweet's song is I'm Suzie Sweet, I'm Penny Pocket (Series 1-4).
  Fact 0.36: Suzie Sweet's song is Suzie's Cooking (Series 3-4).
  Fact 0.37: Penny Pocket is played by Kim Tserkezie.
  Fact 0.38: Penny Pocket's song is I'm Suzie Sweet, I'm Penny Pocket (Series 1-4).
  Fact 0.39: Penny Pocket's song is Sort It (Series 3-4).

Retrieved items:
<item 1>
Balamory
The storyline revolves around simple problems designed to challenge pre schoolers ' minds . Songs ( edit ) There are many songs in Balamory , mainly sung by the
actors . These songs are used consistently throughout the series , when corresponding characters or circumstances are introduced . In the run starting from
September 2004 , new songs were introduced . These songs are longer and take up more time in the episode , so the old versions were still often used . In
addition to the character specific songs mentioned in the `` Characters '' section , there are two general songs : `` What 's the Story in Balamory ? '' is the
show 's theme song , sung over the opening titles and played as an instrumental over the closing credits . `` Which Coloured House Are We Going To ? '' is sung
by the main character of each episode after introducing that episode 's problem . Production ( edit ) Very few episodes feature actors other than the main
characters and the nursery children . Terry Wogan made a guest appearance in one episode ( The Game Show ) as a television director .
</item 1>

<item 2>
Balamory
Penny Pocket is a young woman who uses a wheelchair ; she has a talent for mathematics . Episodes ( edit ) A typical episode follows this general plan : The
opening credits show the town of `` Balamory '' with its brightly coloured houses . Miss Hoolie opens the nursery school , greets the nursery children , and
talks about that day 's weather . One of the other characters ( who will be the main character of that episode ) tells Miss Hoolie about a problem . This
character then sings `` Which Coloured House Are We Going To ? '' , asking the young viewers to guess which of the Balamory characters will be most helpful in
solving the problem . The main character then proceeds to visit several other characters until the problem is solved . Miss Hoolie and the main character
provide a summary of the story at each point , and a final recap at the end of the episode . For example , in one episode Josie Jump has been asked to look
after a neighbour 's rabbit , but does not know what it needs . She is advised to visit PC Plum , who advises her to feed it vegetables .
</item 2>

<item 3>
Balamory
It was also shown on TLC . In Ireland , episodes were shown until recently on weekdays on RTÉ Two 's The Den and continue to be shown on The Den 's replacement
, RTÉjr . Tobermory , Mull , the setting for the fictional town of Balamory Edie McCredie 's bus ( LDV Convoy ) in the Glasgow Museum of Transport Contents (
hide ) 1 Characters 2 Episodes 2.1 Episode guide 3 Audience 4 Songs 5 Production 6 Retirement 7 Awards 8 References 9 External links Characters ( edit ) There
are eight central characters in Balamory . Each character wears a distinctive colour of clothing like pink or yellow , and lives in a house of the same colour
like Archie lives in a pink castle , with the exception of Penny Pocket ( who shares the red house / shop with Suzie Sweet , and wears blue ) and PC Plum ( who
lives in the white house but wears a police uniform ) . Each character also has one or more individual songs .
</item 3>

<item 4>
Balamory
28 -- 30 minutes . Specials Production company ( s ) BBC Scotland Distributor BBC Release Original network CBeebies Picture format 16 : 9 Widescreen Audio
format Stereo Original release 10 February 2002 -- 14 April 2005 External links Website www.bbc.co.uk/cbeebies/shows/balamory Balamory is a Scottish live action
television series on British television ( BBC One , BBC Two and CBeebies ) for pre school children , about a fictional small island community called Balamory in
Scotland . It was produced between 10 February 2002 and 14 April 2005 by BBC Scotland , with 254 episodes made ( including a DVD exclusive Christmas episode ) .
The series was created by Brian Jameson . Balamory was filmed mostly in Tobermory on the Isle of Mull , with the exception of scenes at Archie 's castle filmed
at Fenton Tower in North Berwick , and other scenes , such as Camden and the nursery filmed in Glasgow . A select series of episodes was shown for two brief
seasons in the United States on the Discovery Kids channel . It was also shown on TLC .
</item 4>

<item 5>
Balamory
External links ( edit ) http://www.bbc.co.uk/cbeebies/shows/balamory Press release at the launch of Balamory ) at bbc.co.uk
http://www.bbc.co.uk/pressoffice/pressreleases/stories/2002/08_august/30/balamory.shtml CBeebies - Balamory ) at bbc.co.uk Balamory on IMDb Balamory at TV.com
Retrieved from `` https://en.wikipedia.org/w/index.php?title=Balamory&oldid=807129679 '' Categories : Balamory BBC children 's television programmes Fictional
populated places in Scotland 2002 British television programme debuts 2005 British television programme endings 2000s Scottish television series Musical
television series Preschool education television series Television shows set in Scotland Isle of Mull 2002 establishments in Scotland BBC Scotland television
programmes 2000s British children 's television series Hidden categories : Use dmy dates from March 2014 Articles needing additional references from December
2010 All articles needing additional references Talk Contents About Wikipedia Dansk Polski Edit links This page was last edited on 26 October 2017 , at 02 : 40
.
</item 5>

<item 6>
Balamory
Balamory - wikipedia Balamory Jump to : navigation , search This article needs additional citations for verification . Please help improve this article by
adding citations to reliable sources . Unsourced material may be challenged and removed . ( December 2010 ) ( Learn how and when to remove this template message
) Balamory Created by Brian Jameson Starring Julie Wilson Nimmo Miles Jupp Kasia Haddad Buki Akib Juliet Cadzow Andrew Agnew Rodd Christensen Mary Riggans Kim
Tserkezie Country of origin United Kingdom Original language ( s ) English No. of episodes 254 , including Christmas special . Production Producer ( s ) Brian
Jameson Running time 19 -- 21 minutes . 28 -- 30 minutes .
</item 6>

<item 7>
Balamory
Edie McCredie drives the nursery school bus . She has travelled all over the world , and has a variety of photographs and souvenirs to talk about . She is also
an expert driver and motor mechanic , and occasionally gets exasperated with other road users . Police officer PC Plum aspires to be a master detective , but
Balamory is a crime free village . He spends most of his time watching wildlife and using his detective skills , to figure out what the animals have been doing
. He is also particularly interested in rules , regulations , and safety . Spencer is officially responsible for painting the coloured houses of Balamory , but
also paints pictures and creates music . His painter 's ladder is musical , each rung sounding a different note . Spencer is also the only American to reside in
Balamory and he is black . Suzie Sweet and Penny Pocket together run the village shop and cafe . Suzie Sweet is an older , grandmotherly figure who loves to
tell customers about the items in the shop . Penny Pocket is a young woman who uses a wheelchair ; she has a talent for mathematics .
</item 7>

<item 8>
Balamory
Each character also has one or more individual songs . Character Actor Job House Song ( s ) Miss Hoolie Julie Wilson Nimmo Nursery teacher Green house Nursery
Song : Everybody everyone Day off song : What do you want to do today ( Series 1 -- 2 ) Strike up the band ( Series 3 -- 4 ) Archie the Inventor Miles Jupp
Inventor Pink castle I 'm Archie the Inventor ( Series 1 - 4 ) Great Inventions ,
</item 8>

<item 9>
Balamory
This episode also featured Greg Hemphill ( Julie Wilson Nimmo 's husband in real life ) and John Altman , who plays Nick Cotton in the BBC soap opera EastEnders
. Celebrity chef Keith Floyd appeared in Suzie Sweet 's song `` Suzie 's Cookin ' '' . Occasionally , the shots of children entering the nursery are cut down
for repeats for time reasons . Retirement ( edit ) On 30 April 2005 , the BBC announced that it would not order another season of the series , citing they
wanted to experiment and create more new programme ideas . In that year , all the cast ( except for two of them ) performed a farewell touring show called ``
Balamory Live : Strike up the Band '' , a follow up to the `` Balamory Live : What 's The Story '' tour from 2004 . By the time the programme was retired , many
of the cast were also keen to move on . Miles Jupp , who played Archie , stated that his role caused some problems when he was performing stand up on the
Edinburgh Fringe , due to parents assuming his show was child friendly , and bringing their children to see it .
</item 9>

<item 10>
Balamory
Groovy Solutions ( Series 3 -- 4 ) Josie Jump Buki Akib ( series 1 - 2 ) Kasia Haddad ( Series 3 -- 4 ) Fitness instructor Yellow house Josie Jump is My Name (
Jump a Little Higher ) ( Series 1 - 4 ) Cheer You Up ( Series 3 -- 4 ) Edie McCredie Juliet Cadzow Bus driver Blue garage / house When I Honk My Horn ( Series 1
- 4 ) Let Me Take You on a Journey ( Series 3 -- 4 ) PC Plum Andrew Agnew Policeman Black and white police station I 'm PC Plum ( Series 1 -- 4 ) Follow the
Clue ( Series 3 -- 4 ) Spencer the Painter Rodd Christensen Painter / Musician Orange house Climbing Up My Musical Ladder ( Series 1 -- 4 ) If You Need a Little
Rhythm ( Series 3 -- 4 ) Suzie Sweet Mary Riggans Shop and café workers Red shop I 'm Suzie Sweet ,
</item 10>
=== END ITEM id=0ff65f84339efb6b ===

=== ITEM id=03b46769eb7ed795 ===
Gold sentences and facts:
Sentence 0: The following season , a cancerous desmoid tumor was found in Dravecky 's pitching arm . On October 7 , 1988 , he underwent surgery , which removed half of the deltoid muscle in his pitching arm and froze the humerus bone in an effort to eliminate all of the cancerous cells . Doctors advised Dravecky to wait until 1990 to pitch again , but Dravecky was determined to pitch in 1989 . By July 1989 , he was pitching in the minors , and on August 10 , he made a highly publicized return to the major leagues , pitching eight innings and defeating Cincinnati 4 -- 3 . In his following start , five days later in Montreal , Dravecky pitched three no - hit innings , but in the fifth inning , he felt a tingling sensation in his arm . In the sixth inning , he started off shaky , allowing a home run to the leadoff batter and then hitting the second batter , Andrés Galarraga . Then , on his first pitch to Tim Raines , his humerus bone snapped ; the sound of it breaking could be heard throughout the stadium . Dravecky collapsed on the mound . He 'd suffered a clean break midway between his shoulder and elbow , ending his season .
  Fact 0.0: The following season, a cancerous desmoid tumor was found in Dravecky's pitching arm.
  Fact 0.1: On October 7, 1988, Dravecky underwent surgery.
  Fact 0.2: The surgery removed half of the deltoid muscle in Dravecky's pitching arm.
  Fact 0.3: The surgery froze the humerus bone in Dravecky's pitching arm in an effort to eliminate all of the cancerous cells.
  Fact 0.4: Doctors advised Dravecky to wait until 1990 to pitch again.
  Fact 0.5: Dravecky was determined to pitch in 1989.
  Fact 0.6: By July 1989, Dravecky was pitching in the minors.
  Fact 0.7: On August 10, Dravecky made a highly publicized return to the major leagues.
  Fact 0.8: In that return, Dravecky pitched eight innings and defeated Cincinnati 4-3.
  Fact 0.9: In his following start, five days later in Montreal, Dravecky pitched three no-hit innings.
  Fact 0.10: In the fifth inning of that start, Dravecky felt a tingling sensation in his arm.
  Fact 0.11: In the sixth inning, Dravecky started off shaky.
  Fact 0.12: In the sixth inning, Dravecky allowed a home run to the leadoff batter.
  Fact 0.13: In the sixth inning, Dravecky hit the second batter, Andrés Galarraga.
  Fact 0.14: On his first pitch to Tim Raines, Dravecky's humerus bone snapped.
  Fact 0.15: The sound of the bone breaking could be heard throughout the stadium.
  Fact 0.16: Dravecky collapsed on the mound.
  Fact 0.17: Dravecky suffered a clean break midway between his shoulder and elbow.
  Fact 0.18: The broken bone ended Dravecky's season.

Retrieved items:
<item 1>
2017 World Series
Morrow relieved Darvish and in the process became only the second pitcher to pitch in all seven games of a single World Series , joining Darold Knowles in the
1973 World Series . McCullers lasted only 2 ​ ⁄ innings himself ; he allowed three hits and hit a World Series record four batters . This was the first Game 7
in World Series history where neither starting pitcher got past the third inning . The Dodgers failed to score a run and left eight men on base through five
innings . Ethier , pinch hitting , scored Pederson on a single in the sixth inning for the Dodgers ' only run . They only had one hit in 13 chances with runners
in scoring position in the game . Kershaw pitched four scoreless innings of relief in the game , and in the process , he broke Orel Hershiser 's Dodgers
postseason record with his 33rd strikeout .
</item 1>

<item 2>
Houston Astros
Surgery was done to save his life , but the Astros had lost their ace pitcher after a 10 -- 4 start with a stingy 1.89 ERA . Richard attempted a comeback , but
would never again pitch a big league game . After the loss of Richard and some offensive struggles , the Astros slipped to third place in the division behind
the Dodgers and the Reds . They bounced back to first with a ten - game winning streak , but the Dodgers had regained a two - game lead when they arrived in
Houston on September 9 . The Astros won the first two games of that series and the two teams were tied for the division lead . The Astros held a three - game
lead over the Dodgers with three games left in the season against the Dodgers . The Dodgers swept the series games , forcing a one - game playoff the next day .
The Astros would however win the one - game playoff 7 - 1 , and advance to their first post-season . The team would face the Philadelphia Phillies in the 1980
National League Championship Series . The Phillies sent out Steve Carlton in game one of the NLCS after a six - hour flight the night before .
</item 2>

<item 3>
2017 World Series
Game 1 : The ceremonial first pitch was thrown out by members of former Dodger Jackie Robinson 's family , including his widow Rachel . The game marked the 45th
anniversary of Robinson 's death , and the 2017 season was the 70th anniversary of his breaking of the baseball color line . Keith Williams Jr. , a gospel
singer , performed `` The Star - Spangled Banner '' , the national anthem . Game 2 : Fernando Valenzuela threw out the ceremonial first pitch to Steve Yeager ;
both were introduced by retired Dodgers broadcaster Vin Scully . Country music 's Brad Paisley performed the national anthem . Game 3 : Houston Texans defensive
end J.J. Watt , who had raised $37 million for Hurricane Harvey victims , threw out the first pitch . The ball was given to him by Astros ' Hall of Famer Craig
Biggio . The national anthem was performed by Texas Air National Guard Master Sergeant Promise Harris . Game 4 : Hailey Dawson , a seven - year - old girl from
Nevada , threw out the ceremonial first pitch using a 3D printed hand . The national anthem was performed by the Houston Police Department Quartet .
</item 3>

<item 4>
Dave Dravecky
Crisis and comeback ( edit ) The following season , a cancerous desmoid tumor was found in Dravecky 's pitching arm . On October 7 , 1988 , he underwent surgery
, which removed half of the deltoid muscle in his pitching arm and froze the humerus bone in an effort to eliminate all of the cancerous cells . Doctors advised
Dravecky to wait until 1990 to pitch again , but Dravecky was determined to pitch in 1989 . By July 1989 , he was pitching in the minors , and on August 10 , he
made a highly publicized return to the major leagues , pitching eight innings and defeating Cincinnati 4 -- 3 . In his following start , five days later in
Montreal , Dravecky pitched three no - hit innings , but in the fifth inning , he felt a tingling sensation in his arm . In the sixth inning , he started off
shaky , allowing a home run to the leadoff batter and then hitting the second batter , Andrés Galarraga .
</item 4>

<item 5>
Dave Dravecky
Both retrieved from Google News May 23 , 2013 . Jump up ^ Dravecky , Dave ( 2004 ) . Called Up : Stories of Life and Faith from the Great Game of Baseball .
Zondervan . p. 162 . ISBN 031087159X . Jump up ^ Thomas , Robert ( August 9 , 1989 ) . `` Dravecky is Back on Center Stage '' . New York Times . Retrieved June
24 , 2010 . ^ Jump up to : Thomas , Robert ( August 17 , 1989 ) . `` Dravecky Was Told He Risked Fracture '' . New York Times . Retrieved June 24 , 2010 . Jump
up ^ Associated Press ( June 19 , 1991 ) . `` SPORTS PEOPLE : BASEBALL ; Dravecky 's Left Arm Amputated , Giants Say '' . New York Times . Retrieved June 24 ,
2010 . Jump up ^ Associated Press ( October 10 , 1989 ) . `` Dravecky Hurt Again '' .
</item 5>

<item 6>
Houston Astros
McMullen showed the city of Houston that he too wanted a winning team , signing nearby Alvin , Texas native Nolan Ryan to the first million - dollar - a-year
deal . Ryan had four career no - hitters already and had struck out 383 in one season . 1980 : on the brink ( edit ) Joe Morgan returned in 1980 . Now back in
Houston with two MVP awards and two World Series rings , Morgan wanted to help make the Astros a pennant winner . The 1980 pitching staff was one of the best
Houston ever had , with the fastball of Ryan , the knuckleball of Joe Niekro and the terrifying 6 ft 8 in frame of J.R. Richard . Teams felt lucky to face Ken
Forsch , who was a double - digit winner in the previous two seasons . Richard became the first Astros pitcher to start an All - Star game . After a medical
examination three days later , Richard was told to rest his arm and he collapsed during a July 30 workout . He had suffered a stroke after a blood clot in the
arm apparently moved to his neck and cut off blood flow to the brain .
</item 6>

<item 7>
Dave Dravecky
Fred Anderson Johnny Antonelli Jesse Barnes Virgil Barnes Jim Barr Larry Benton Vida Blue Tom Bradley Madison Bumgarner John Burkett Matt Cain Mark Davis Phil
Douglas Dave Dravecky Shawn Estes Freddie Fitzsimmons Mark Gardner Rubén Gómez Harry Gumbert Atlee Hammaker Liván Hernández Al Holland Carl Hubbell Larry Jansen
Bob Knepper Mike Krukow Mark Leiter Tim Lincecum Bill Lohrman Sal Maglie Juan Marichal Rube Marquard Christy Mathewson Joe McGinnity Hugh McQuillan Cliff Melton
John Montefusco Terry Mulholland Art Nehf Gaylord Perry Rick Reuschel Kirk Rueter Rosy Ryan Jack Sanford Jason Schmidt Hal Schumacher Bill Swift Dummy Taylor
Jeff Tesreau Bill Voiselle Bill Walker Barry Zito VIAF : 77929919 LCCN : n89666209 Retrieved from `` https://en.wikipedia.org/w/index.
</item 7>

<item 8>
Dave Dravecky
1980 : Brett 1981 : Bench 1982 : Thornton 1983 : Knight 1984 : Robinson 1985 : Reuschel 1986 : Leonard 1987 : Molitor 1988 : Oester 1989 : Dravecky 1990 : Bream
1991 : Wegman 1992 : Lansford 1993 : Olerud 1994 : Dawson 1995 : Abbott 1996 : Vizquel 1997 : Davis 1998 : Cone 1999 : Casey 2000 : Giambi 2001 : Schilling 2002
: Salmon 2003 : Moyer 2004 : Hoffman 2005 : Biggio 2006 : Loretta 2007 : Sweeney 2008 : Lester 2009 : Teahen 2010 : Hudson 2011 : Butler 2012 : Zito 2013 :
Ibañez 2014 : Gordon 2015 : Wainwright 2016 : McGowan New York / San Francisco Giants Opening Day starting pitchers Red Ames Fred Anderson Johnny Antonelli
Jesse Barnes Virgil Barnes Jim Barr Larry Benton Vida Blue Tom Bradley Madison
</item 8>

<item 9>
Dave Dravecky
Then , on his first pitch to Tim Raines , his humerus bone snapped ; the sound of it breaking could be heard throughout the stadium . Dravecky collapsed on the
mound . He 'd suffered a clean break midway between his shoulder and elbow , ending his season . The Giants won the National League pennant in 1989 ( defeating
the Chicago Cubs in the NLCS in five games ) , and in the post-game celebration , Dravecky 's arm was broken a second time when he was running out to the mound
to celebrate . A doctor examining Dravecky 's x-rays noticed a mass in his arm , which turned out to be malignant . Dravecky 's cancer had returned , ending his
career . Eighteen days later , Dravecky retired from baseball with a 64 -- 57 record with 558 strikeouts and a 3.13 ERA in 1,062 2⁄3 innings . He won the 1989
Willie Mac Award honoring his spirit and leadership .
</item 9>

<item 10>
2017 World Series
This was the first time since the 2001 World Series and 2002 World Series that back - to - back Fall Classics had a Game 7 . The starting pitchers for this game
were the same as in the third game : McCullers and Darvish . Springer doubled to open the game and scored the first run on an error by Bellinger , which allowed
Bregman to reach second base . Bregman stole third base and scored on an Altuve groundout . McCann scored the next inning on a groundout hit by the pitcher
McCullers . Springer hit a two - run home run , his fifth of the series , tying Reggie Jackson and Chase Utley for most home runs in a single World Series and
setting a new record with 29 total bases in any postseason series . As in Game 3 , Darvish lasted only ​ 1 ⁄ innings ( tying the shortest outing of his career )
and became the third pitcher with two starts of less than two innings in a World Series , and the first since Art Ditmar in the 1960 World Series .
</item 10>
=== END ITEM id=03b46769eb7ed795 ===
