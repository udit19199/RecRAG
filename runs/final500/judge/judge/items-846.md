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

=== ITEM id=489af81af966a47f ===
Gold sentences and facts:
Sentence 0: Cousins Bo ( Seann William Scott ) , Luke ( Johnny Knoxville ) , and Daisy Duke ( Jessica Simpson ) run a moonshine business for their Uncle Jesse ( Willie Nelson ) in Hazzard County , Georgia . The cousins ' primary mode of transportation is an orange 1969 Dodge Charger that the boys affectionately refer to as the `` General Lee '' . Along the way , the family is tormented by corrupt Hazzard County Commissioner Jefferson Davis Hogg , widely known as `` Boss Hogg '' ( Burt Reynolds ) , and his willing but dimwitted henchman , Sheriff Rosco P. Coltrane ( M.C. Gainey ) .
  Fact 0.0: Bo, played by Seann William Scott, is a cousin
  Fact 0.1: Luke, played by Johnny Knoxville, is a cousin
  Fact 0.2: Daisy Duke, played by Jessica Simpson, is a cousin
  Fact 0.3: Bo, Luke, and Daisy Duke run a moonshine business
  Fact 0.4: Bo, Luke, and Daisy Duke run the moonshine business for their Uncle Jesse
  Fact 0.5: Uncle Jesse, played by Willie Nelson, is their uncle
  Fact 0.6: The moonshine business is in Hazzard County, Georgia
  Fact 0.7: The cousins' primary mode of transportation is an orange 1969 Dodge Charger
  Fact 0.8: The boys affectionately refer to the orange 1969 Dodge Charger as the 'General Lee'
  Fact 0.9: The family is tormented by corrupt Hazzard County Commissioner Jefferson Davis Hogg
  Fact 0.10: Jefferson Davis Hogg is widely known as 'Boss Hogg'
  Fact 0.11: Boss Hogg is played by Burt Reynolds
  Fact 0.12: Boss Hogg has a willing but dimwitted henchman, Sheriff Rosco P. Coltrane
  Fact 0.13: Sheriff Rosco P. Coltrane is played by M.C. Gainey
  Fact 0.14: Sheriff Rosco P. Coltrane is willing but dimwitted
Sentence 1: Jessica Simpson as Daisy Duke
  Fact 1.0: Jessica Simpson plays Daisy Duke

Retrieved items:
<item 1>
The Dukes of Hazzard (film)
Gainey as Sheriff Rosco P. Coltrane Michael Weston as Deputy Enos Strate Lynda Carter as Pauline James Roday as Billy Prickett Kevin Heffernan as Derek `` Sheev
'' Sheevington Nikki Griffin as Katie - Lynn Johnson Jacqui Maxwell as Annette Alice Greczyn as Laurie Pullman Junior Brown as The Balladeer ( narrator ) Joe
Don Baker as Governor Jim Applewhite Barry Corbin as Bill Pullman Andrew Prine as Angry Man Randy Orton as Bo Duke ( Stunt double ) Cameos ( edit ) All five
members of the comedy film troupe Broken Lizard make appearances in the film , classified as cameos , except for Kevin Heffernan , who had a larger speaking
role ( Sheev ) . Broken Lizard cameos Steve Lemme appears as Jimmy Pullman , the son of Bill Pullman , in an early car chase scene in which he accidentally
shoots the inside of his father 's truck . Jay Chandrasekhar and Erik Stolhanske reprise their roles as Ramathorn and Rabbit from the Broken Lizard comedy ,
Super Troopers .
</item 1>

<item 2>
The Dukes of Hazzard (film)
( 1997 ) The Dukes of Hazzard : Hazzard in Hollywood ( 2000 ) Remake films The Dukes of Hazzard ( 2005 ) The Dukes of Hazzard : The Beginning ( 2007 ) Video
games The Dukes of Hazzard Racing for Home Daisy Dukes It Out Return of the General Lee Spin - offs Enos The Dukes See also The General Lee Boar 's Nest Jerry
Rushing `` Theme from The Dukes of Hazzard ( Good Ol ' Boys ) '' Moonrunners Hazard Run Films directed by Jay Chandrasekhar Puddle Cruiser ( 1996 ) Super
Troopers ( 2001 ) Club Dread ( 2004 ) The Dukes of Hazzard ( 2005 ) Beerfest ( 2006 ) Broken Lizard Stands Up ( 2010 ) The Babymakers ( 2012 ) Super Troopers 2
( 2018 ) VIAF : 185752175 LCCN : no2005089888 Retrieved from `` https://en.wikipedia.org/w/index.
</item 2>

<item 3>
The Dukes of Hazzard (film)
Controversy ( edit ) Before the release of this film , Warner Bros. reportedly paid $17.5 million to the producer of Moonrunners , the movie that inspired the
television series . This was soon followed by a claim from screenwriter Gy Waldron . The Hollywood Reporter reported that James Best , who portrayed Rosco P.
Coltrane in the original series , filed suit in late July 2011 over royalties he was contracted to receive over spinoffs that `` used his identity '' . Ben
Jones , who played Cooter Davenport in the original series , criticized the film for its emphasis on sexual content , suggesting that the original series was
more family - oriented and not as sexualized . He called for fans of the television series to boycott the film `` unless they clean it up before the August 5th
release date . '' Some have countered that the original series also contained sexual themes , primarily Catherine Bach 's ( Daisy Duke ) much - displayed ``
short shorts '' ( which have become so ubiquitous in American culture that skimpy blue jean cutoff shorts are now often simply called `` Daisy Dukes '' ) .
</item 3>

<item 4>
The Dukes of Hazzard (film)
Worst Picture - lost to Dirty Love Worst Director ( Jay Chandrasekhar ) - lost to John Mallory Asher for Dirty Love Worst Screenplay ( John O'Brien ) - lost to
Jenny McCarthy for Dirty Love Worst Supporting Actor ( Burt Reynolds ) - lost to Hayden Christensen in Star Wars : Episode III -- Revenge of the Sith Worst
Supporting Actress ( Jessica Simpson ) - lost to Paris Hilton in House of Wax Worst Screen Couple ( Jessica Simpson and her Daisy Dukes ) - lost to Nicole
Kidman and Will Ferrell in Bewitched Worst Remake or Sequel - lost to Son of the Mask At the People 's Choice Awards , Simpson won the `` Favorite Song from a
Movie '' award for her cover of Nancy Sinatra 's `` These Boots Are Made For Walkin ' '' . The film was nominated for two MTV Movie Awards , including Best On -
Screen Team ( Johnny Knoxville , Seann William Scott , and Jessica Simpson ) , and Sexiest Performance ( Jessica Simpson ) . Simpson won the Choice Breakout
Female award for her role in the film at the Teen Choice Awards . Controversy ( edit ) Before the release of this film , Warner Bros.
</item 4>

<item 5>
The Dukes of Hazzard (film)
The Dukes of Hazzard ( film ) - wikipedia The Dukes of Hazzard ( film ) Jump to : navigation , search The Dukes of Hazzard Theatrical release poster Directed by
Jay Chandrasekhar Produced by Bill Gerber Screenplay by John O'Brien Story by John O'Brien Jonathan L. Davis Based on Characters by Gy Waldron Starring Johnny
Knoxville Seann William Scott Jessica Simpson Burt Reynolds Willie Nelson Narrated by Junior Brown Music by Nathan Barr Cinematography Lawrence Sher Edited by
Lee Haxall Myron I. Kerstein Production company Village Roadshow Pictures Distributed by Warner Bros. Pictures Release date August 5 , 2005 ( 2005 - 08 - 05 )
Running time 104 minutes Country United States Language English Budget $50 million Box office $111.1 million The Dukes of Hazzard is a 2005 American buddy
comedy road film based on the television series , The Dukes of Hazzard . The film was directed by Jay Chandrasekhar and released on August 5 , 2005 , by Warner
Bros. Pictures .
</item 5>

<item 6>
The Dukes of Hazzard (film)
The cousins ' primary mode of transportation is an orange 1969 Dodge Charger that the boys affectionately refer to as the `` General Lee '' . Along the way ,
the family is tormented by corrupt Hazzard County Commissioner Jefferson Davis Hogg , widely known as `` Boss Hogg '' ( Burt Reynolds ) , and his willing but
dimwitted henchman , Sheriff Rosco P. Coltrane ( M.C. Gainey ) . After Rosco has the General Lee impounded after Bo and Luke 's attempt to run away from a
daughter of one their many moonshine customers , Billy Prickett ( James Roday ) , a famous stock - car driver , enters Hazzard to participate in the rally .
Meanwhile , Rosco plants a fake moonshine still ( `` ' cause he 's too dumb to find the real one '' ) in Uncle Jesse 's barn and seizes the Duke property in the
interest of eminent domain for Boss Hogg , forcing the family to temporarily reside with neighbor and Uncle Jesse 's love - interest , Pauline ( Lynda Carter )
.
</item 6>

<item 7>
The Dukes of Hazzard (film)
In a film review , a New York Daily News entertainment columnist said the movie 's sex humor is `` cruder '' than the TV series , but that it is `` nearly
identical to the TV series in ... its ogling of the posterior of cousin Daisy Duke . '' Although initially he commented that he enjoyed the new style of
relationship between the movie versions of Bo and Luke , John Schneider , who played Bo Duke in the original series , was later asked if he saw the film and
said : `` My gosh ... it was terrible ! It was n't Dukes . It was true to whatever it was ; I just do n't know what that was ! '' Soundtrack ( edit ) Main
article : These Boots Are Made for Walkin ' § Jessica Simpson version Jessica Simpson recorded her own version of `` These Boots Are Made for Walkin ' '' ( and
added her own lyrics ) for the soundtrack to the film The Dukes of Hazzard ( 2005 ) . Performed from the point of view of her character in the movie , Simpson
's cover was co-produced by Jimmy Jam and Terry Lewis , and was released as the soundtrack 's first single in 2005 .
</item 7>

<item 8>
The Dukes of Hazzard (film)
Pictures . As in the television series , the film depicts the adventures of cousins Bo , Luke , and Daisy , and their Uncle Jesse , as they outfox crooked
Hazzard County Commissioner Boss Hogg and Sheriff Rosco P. Coltrane . The film was the acting debut of pop singer Jessica Simpson . While financially successful
, the film met with negative reviews from critics . The film was followed by a direct - to - video prequel titled The Dukes of Hazzard : The Beginning ( 2007 )
. Contents ( hide ) 1 Plot 2 Cast 2.1 Cameos 3 Production 3.1 Stunts 3.2 Locations 4 Reception 4.1 Box office 4.2 Critical reception 4.3 Awards 5 Controversy 6
Soundtrack 7 Unrated Cut 8 Prequel 9 References 10 External links Plot ( edit ) Cousins Bo ( Seann William Scott ) , Luke ( Johnny Knoxville ) , and Daisy Duke
( Jessica Simpson ) run a moonshine business for their Uncle Jesse ( Willie Nelson ) in Hazzard County , Georgia .
</item 8>

<item 9>
The Dukes of Hazzard (film)
The final scene shows a cook - out at the Dukes ' house where Pauline convinces Uncle Jesse , who could not be found because he was `` using the meat smoker ''
, to get up and play the television series ' main theme . Bo and Luke are romantically involved with the girls in the General Lee when they are caught by Luke
's other love - interest Laurie Pullman ( Alice Greczyn ) from the introduction of the film , who proceeds to chase them with a shotgun as they drive away .
During the credits , outtakes from the production of the film are shown . Cast ( edit ) Knoxville and Scott at the premiere Johnny Knoxville as Luke Duke Seann
William Scott as Bo Duke Jessica Simpson as Daisy Duke Burt Reynolds as Boss Hogg Willie Nelson as Uncle Jesse Duke David Koechner as Cooter Davenport M.C.
</item 9>

<item 10>
The Dukes of Hazzard (film)
www.usatoday.com . January 5 , 2006 . Retrieved 2008 - 07 - 31 . Jump up ^ Wurm , Gerald . `` Dukes of Hazzard , The ( Comparison : Theatrical Cut - Unrated
Edition ) - Movie-Censorship.com '' . www.movie-censorship.com . External links ( edit ) Wikimedia Commons has media related to The Dukes of Hazzard ( film ) .
Wikiquote has quotations related to : The_Dukes_of_Hazzard_ ( film ) The Dukes of Hazzard on IMDb The Dukes of Hazzard at AllMovie The Dukes of Hazzard at Box
Office Mojo The Dukes of Hazzard at Rotten Tomatoes The Dukes of Hazzard at HazzardNet.com The Dukes of Hazzard Episodes Characters Bo Duke Luke Duke Daisy Duke
Sheriff Rosco P. Coltrane Boss Hogg Television films The Dukes of Hazzard : Reunion !
</item 10>
=== END ITEM id=489af81af966a47f ===

=== ITEM id=1e86517374642d19 ===
Gold sentences and facts:
Sentence 0: Year Film Nominees 2001 ( 74th ) Shrek Aron Warner Jimmy Neutron : Boy Genius Steve Oedekerk and John A. Davis Monsters , Inc . Pete Docter and John Lasseter 2002 ( 75th ) Spirited Away Hayao Miyazaki Ice Age Chris Wedge Lilo & Stitch Chris Sanders Spirit : Stallion of the Cimarron Jeffrey Katzenberg Treasure Planet Ron Clements 2003 ( 76th ) Finding Nemo Andrew Stanton Brother Bear Aaron Blaise and Robert Walker The Triplets of Belleville Sylvain Chomet ( 77th ) The Incredibles Brad Bird Shark Tale Bill Damaschke Shrek 2 Andrew Adamson 2005 ( 78th ) Wallace & Gromit : The Curse of the Were - Rabbit Nick Park and Steve Box Howl 's Moving Castle Hayao Miyazaki Corpse Bride Mike Johnson and Tim Burton 2006 ( 79th ) Happy Feet George Miller Cars John Lasseter Monster House Gil Kenan 2007 ( 80th ) Ratatouille Brad Bird Persepolis Marjane Satrapi and Vincent Paronnaud Surf 's Up Ash Brannon and Chris Buck 2008 ( 81st ) WALL - E Andrew Stanton Bolt Chris Williams and Byron Howard Kung Fu Panda John Stevenson and Mark Osborne 2009 ( 82nd ) Up Pete Docter Coraline Henry Selick Fantastic Mr. Fox Wes Anderson The Princess and the Frog John Musker and Ron Clements The Secret of Kells Tomm Moore
  Fact 0.0: At the 2001 (74th) list, Shrek was nominated with Aron Warner as nominee.
  Fact 0.1: At the 2001 (74th) list, Jimmy Neutron: Boy Genius was nominated with Steve Oedekerk and John A. Davis as nominees.
  Fact 0.2: At the 2001 (74th) list, Monsters, Inc. was nominated with Pete Docter and John Lasseter as nominees.
  Fact 0.3: At the 2002 (75th) list, Spirited Away was nominated with Hayao Miyazaki as nominee.
  Fact 0.4: At the 2002 (75th) list, Ice Age was nominated with Chris Wedge as nominee.
  Fact 0.5: At the 2002 (75th) list, Lilo & Stitch was nominated with Chris Sanders as nominee.
  Fact 0.6: At the 2002 (75th) list, Spirit: Stallion of the Cimarron was nominated with Jeffrey Katzenberg as nominee.
  Fact 0.7: At the 2002 (75th) list, Treasure Planet was nominated with Ron Clements as nominee.
  Fact 0.8: At the 2003 (76th) list, Finding Nemo was nominated with Andrew Stanton as nominee.
  Fact 0.9: At the 2003 (76th) list, Brother Bear was nominated with Aaron Blaise and Robert Walker as nominees.
  Fact 0.10: At the 2003 (76th) list, The Triplets of Belleville was nominated with Sylvain Chomet as nominee.
  Fact 0.11: At the (77th) list, The Incredibles was nominated with Brad Bird as nominee.
  Fact 0.12: At the (77th) list, Shark Tale was nominated with Bill Damaschke as nominee.
  Fact 0.13: At the (77th) list, Shrek 2 was nominated with Andrew Adamson as nominee.
  Fact 0.14: At the 2005 (78th) list, Wallace & Gromit: The Curse of the Were-Rabbit was nominated with Nick Park and Steve Box as nominees.
  Fact 0.15: At the 2005 (78th) list, Howl's Moving Castle was nominated with Hayao Miyazaki as nominee.
  Fact 0.16: At the 2005 (78th) list, Corpse Bride was nominated with Mike Johnson and Tim Burton as nominees.
  Fact 0.17: At the 2006 (79th) list, Happy Feet was nominated with George Miller as nominee.
  Fact 0.18: At the 2006 (79th) list, Cars was nominated with John Lasseter as nominee.
  Fact 0.19: At the 2006 (79th) list, Monster House was nominated with Gil Kenan as nominee.
  Fact 0.20: At the 2007 (80th) list, Ratatouille was nominated with Brad Bird as nominee.
  Fact 0.21: At the 2007 (80th) list, Persepolis was nominated with Marjane Satrapi and Vincent Paronnaud as nominees.
  Fact 0.22: At the 2007 (80th) list, Surf's Up was nominated with Ash Brannon and Chris Buck as nominees.
  Fact 0.23: At the 2008 (81st) list, WALL-E was nominated with Andrew Stanton as nominee.
  Fact 0.24: At the 2008 (81st) list, Bolt was nominated with Chris Williams and Byron Howard as nominees.
  Fact 0.25: At the 2008 (81st) list, Kung Fu Panda was nominated with John Stevenson and Mark Osborne as nominees.
  Fact 0.26: At the 2009 (82nd) list, Up was nominated with Pete Docter as nominee.
  Fact 0.27: At the 2009 (82nd) list, Coraline was nominated with Henry Selick as nominee.
  Fact 0.28: At the 2009 (82nd) list, Fantastic Mr. Fox was nominated with Wes Anderson as nominee.
  Fact 0.29: At the 2009 (82nd) list, The Princess and the Frog was nominated with John Musker and Ron Clements as nominees.
  Fact 0.30: At the 2009 (82nd) list, The Secret of Kells was nominated with Tomm Moore as nominee.

Retrieved items:
<item 1>
Academy Award for Best Animated Feature
This could be the producer , the director , or both . For the 76th Academy Awards in 2004 , only the director ( s ) of the film received the nomination . For
the 86th Academy Awards ten years later , this was amended to include one producer and up to two directors . Winners and nominations ( edit ) 2000s ( edit )
Year Film Nominees 2001 ( 74th ) Shrek Aron Warner Jimmy Neutron : Boy Genius Steve Oedekerk and John A. Davis Monsters , Inc . Pete Docter and John Lasseter
2002 ( 75th ) Spirited Away Hayao Miyazaki Ice Age Chris Wedge Lilo & Stitch Chris Sanders Spirit : Stallion of the Cimarron Jeffrey Katzenberg Treasure Planet
Ron Clements 2003 ( 76th ) Finding Nemo Andrew Stanton Brother Bear Aaron Blaise and Robert Walker The Triplets of Belleville Sylvain Chomet ( 77th ) The
Incredibles Brad Bird Shark Tale Bill Damaschke Shrek 2 Andrew Adamson 2005 ( 78th ) Wallace & Gromit : The Curse of the Were -
</item 1>

<item 2>
Academy Award for Best Animated Feature
3 Les Armateurs 3.3. 4 Cartoon Saloon 3.3. 5 Other films 4 Foreign language films 5 Events 6 See also 7 External links History ( edit ) For much of the Academy
Awards ' history , AMPAS was resistant to the idea of a regular Oscar for animated features considering there were simply too few produced to justify such
consideration . Instead , the Academy occasionally bestowed special Oscars for exceptional productions , usually for Walt Disney Pictures , such as for Snow
White and the Seven Dwarfs in 1938 , and the Special Achievement Academy Award for the live action / animated hybrid Who Framed Roger Rabbit in 1989 and Toy
Story in 1996 . In fact , prior to the creation of the award , only one animated film received a Best Picture nomination : 1991 's Beauty and the Beast , also
by Walt Disney Pictures . By 2001 , the rise of sustained competitors to Disney in the feature animated film market , such as DreamWorks Animation ( ironically
, founded by former Disney executive Jeffrey Katzenberg ) , created an increase of film releases of significant annual number enough for AMPAS to reconsider .
</item 2>

<item 3>
Academy Award for Best Animated Feature
The Academy Award for Best Animated Feature was first awarded in 2002 for films made in 2001 . The entire AMPAS membership has been eligible to choose the
winner since the award 's inception . If there are sixteen or more films submitted for the category , the winner is voted from a shortlist of five films , which
has happened eight times , otherwise there will only be three films on the shortlist . Additionally , eight eligible animated features must have been
theatrically released in Los Angeles County within the calendar year for this category to be activated . Animated films can be nominated for other categories ,
but have rarely been so ; Beauty and the Beast ( 1991 ) was the first animated film ever nominated for Best Picture . Up ( 2009 ) and Toy Story 3 ( 2010 ) also
received Best Picture nominations after the Academy expanded the number of nominees from five to ten . Waltz with Bashir ( 2008 ) is the only animated film ever
nominated for Best Foreign Language Film ( though it did not receive a nomination for Best Animated Feature ) .
</item 3>

<item 4>
Academy Award for Best Animated Feature
Ron Clements , and Osnat Shurer My Life as a Zucchini Claude Barras and Max Karli The Red Turtle Michaël Dudok de Wit and Toshio Suzuki 2017 ( 90th ) The Boss
Baby Tom McGrath and Ramsey Ann Naito The Breadwinner Nora Twomey and Anthony Leo Coco Lee Unkrich and Darla K. Anderson Ferdinand Carlos Saldanha Loving
Vincent Dorota Kobiela , Hugh Welchman , and Ivan Mactaggart Multiple wins ( edit ) 2 wins Brad Bird Pete Docter Andrew Stanton 1 win Chris Buck Byron Howard
Hayao Miyazaki Rich Moore Lee Unkrich Chris Williams Multiple nominations ( edit ) 3 nominations Ron Clements Pete Docter Hayao Miyazaki Chris Sanders 2
nominations Brad Bird Chris Buck Tim Burton Sylvain Chomet Dean DeBlois Byron Howard Travis Knight John Lasseter Rich Moore Tomm Moore John Musker Yoshiaki
Nishimura Andrew Stanton Toshio Suzuki Lee Unkrich Chris Williams Studio breakdown ( edit ) Studio Wins Nominations Films Pixar 8 11 Monsters , Inc.
</item 4>

<item 5>
Academy Award for Best Animated Feature
Academy award for Best animated Feature - wikipedia Academy award for Best animated Feature Jump to : navigation , search Academy Award for Best Animated
Feature Awarded for The best animated film with a running time of more than 40 minutes , a significant number of the major characters animated , and at least 75
percent of the picture 's running time including animation . Country United States Presented by Academy of Motion Picture Arts and Sciences ( AMPAS ) First
awarded 2001 ( for Shrek ) Currently held by Zootopia ( 2016 ) Website oscars.org The Academy Awards are given each year by the Academy of Motion Picture Arts
and Sciences ( AMPAS or the Academy ) for the best films and achievements of the previous year . The Academy Award for Best Animated Feature is given each year
for animated films . An animated feature is defined by the Academy as a film with a running time of more than 40 minutes in which characters ' performances are
created using a frame - by - frame technique , a significant number of the major characters are animated , and animation figures in no less than 75 percent of
the running time .
</item 5>

<item 6>
Academy Award for Best Animated Feature
In 2009 , when the nominee slots for Best Picture were doubled to ten , Up was nominated for both Best Animated Feature and Best Picture at the 82nd Academy
Awards , the first film to do so since the creation of the Animated Feature category . This feat was repeated the following year by Toy Story 3 . Since 2010
onward , with the increasing competitiveness of the Animated Feature category , Pixar ( a perennial nominee ) did not receive nominations for several recent
films considering the studio has released films of more mixed critical reaction and box office earnings , while Pixar 's sister studio Disney Animation won
their first three awards . In 2010 , the Academy enacted a new rule regarding the performance capture technique employed in films such as Disney 's A Christmas
Carol from Robert Zemeckis and The Adventures of Tintin from Steven Spielberg , and how they might not be eligible in this category in the future . This rule
was possibly made to prevent nominations of live - action films that rely heavily on motion capture , such as James Cameron 's Avatar . When the category was
first instated , the nomination went to the person ( s ) most involved in creating the winning film .
</item 6>

<item 7>
Academy Award for Best Animated Feature
2005 ( 78th ) Wallace & Gromit : The Curse of the Were - Rabbit Nick Park and Steve Box Howl 's Moving Castle Hayao Miyazaki Corpse Bride Mike Johnson and Tim
Burton 2006 ( 79th ) Happy Feet George Miller Cars John Lasseter Monster House Gil Kenan 2007 ( 80th ) Ratatouille Brad Bird Persepolis Marjane Satrapi and
Vincent Paronnaud Surf 's Up Ash Brannon and Chris Buck 2008 ( 81st ) WALL - E Andrew Stanton Bolt Chris Williams and Byron Howard Kung Fu Panda John Stevenson
and Mark Osborne 2009 ( 82nd ) Up Pete Docter Coraline Henry Selick Fantastic Mr.
</item 7>

<item 8>
Academy Award for Best Animated Feature
( 82nd ) Up Pete Docter Coraline Henry Selick Fantastic Mr. Fox Wes Anderson The Princess and the Frog John Musker and Ron Clements The Secret of Kells Tomm
Moore 2010s ( edit ) Year Film Nominees ( 83rd ) Toy Story 3 Lee Unkrich How to Train Your Dragon Chris Sanders and Dean DeBlois The Illusionist Sylvain Chomet
2011 ( 84th ) Rango Gore Verbinski A Cat in Paris Alain Gagnol and Jean - Loup Felicioli Chico & Rita Fernando Trueba and Javier Mariscal Kung Fu Panda 2
Jennifer Yuh Nelson Puss in Boots Chris Miller 2012 ( 85th ) Brave Mark Andrews and Brenda Chapman Frankenweenie Tim Burton ParaNorman Sam Fell and Chris Butler
The Pirates ! Band of Misfits Peter Lord Wreck - It Ralph Rich Moore 2013 ( 86th ) Frozen Chris Buck , Jennifer Lee , and Peter Del Vecho The Croods Chris
Sanders , Kirk DeMicco , and Kristine Belson Despicable Me 2 Pierre Coffin , Chris Renaud ,
</item 8>

<item 9>
Academy Award for Best Animated Feature
The Academy Award for Best Animated Feature was first given out at the 74th Academy Awards , held on March 24 , 2002 . The Academy included a rule that stated
that the award would not be presented in a year in which fewer than eight eligible films opened in theaters . People in the animation industry and fans
expressed hope that the prestige from this award and the resulting boost to the box office would encourage the increased production of animated features . Some
members and fans have criticized the award , however , saying it is only intended to prevent animated films from having a chance of winning Best Picture .
DreamWorks had advertised heavily during the holiday 2001 season for Shrek , but was disappointed when the rumored Best Picture nomination did not materialize ,
though it was nominated for and ended up winning the inaugural Best Animated Feature award. . the Shrek franchise to date is most succsfull franchise with one
win and three nomtied .
</item 9>

<item 10>
Academy Award for Best Animated Feature
Book Portal Academy Award for Best Animated Feature Shrek -- Aron Warner ( 2001 ) Spirited Away -- Hayao Miyazaki ( 2002 ) Finding Nemo -- Andrew Stanton ( 2003
) The Incredibles -- Brad Bird ( 2004 ) Wallace & Gromit : The Curse of the Were - Rabbit -- Nick Park and Steve Box ( 2005 ) Happy Feet -- George Miller ( 2006
) Ratatouille -- Brad Bird ( 2007 ) WALL - E -- Andrew Stanton ( 2008 ) Up -- Pete Docter ( 2009 ) Toy Story 3 -- Lee Unkrich ( 2010 ) Rango -- Gore Verbinski (
2011 ) Brave -- Mark Andrews and Brenda Chapman ( 2012 ) Frozen -- Chris Buck , Jennifer Lee , and Peter Del Vecho ( 2013 ) Big Hero 6 -- Don Hall , Chris
Williams , and Roy Conli ( 2014 ) Inside Out -- Pete Docter and Jonas Rivera ( 2015 ) Zootopia -- Byron Howard , Rich Moore , and Clark Spencer ( 2016 )
Retrieved from `` https://en.wikipedia.org/w/index.
</item 10>
=== END ITEM id=1e86517374642d19 ===

=== ITEM id=d5e512aa0e3eee64 ===
Gold sentences and facts:
Sentence 0: Large denominations of United States currency greater than $100 were circulated by the United States Treasury until 1969 . Since then , U.S. dollar banknotes have only been issued in seven denominations : $1 , $2 , $5 , $10 , $20 , $50 , and $100 .
  Fact 0.0: Large denominations of United States currency greater than $100 were circulated by the United States Treasury until 1969.
  Fact 0.1: Since 1969, U.S. dollar banknotes have only been issued in seven denominations.
  Fact 0.2: The seven denominations of U.S. dollar banknotes issued since 1969 are $1, $2, $5, $10, $20, $50, and $100.

Retrieved items:
<item 1>
Large denominations of United States currency
, banknotes with a face value of $500 or higher ) had been used in the United States since the late 18th century . The first $500 note was issued by the
Province of North Carolina , authorized by legislation dated May 10 , 1780 . Virginia quickly followed suit and authorized the printing of $500 and $1,000 notes
on October 16 , 1780 and $2,000 notes on May 7 , 1781 . High - denomination treasury notes were issued , for example during the War of 1812 ( $1,000 notes
authorized by an act dated June 30 , 1812 ) . During the American Civil War Confederate currency included $500 and $1,000 notes . During the Federal banknote
issuing period ( 1861 to present ) , the earliest high - denomination notes included three - year Interest - bearing notes of $500 , $1,000 , and $5,000 ,
authorized by Congress on July 17 , 1861 . In total , 11 different types of U.S. currency were issued in high - denomination notes across nearly 20 different
series dates .
</item 1>

<item 2>
Large denominations of United States currency
( 1995 ) . United States Large Size National Bank Notes . Society of Paper Money Collectors , Inc . ISBN 0 - 9648774 - 1 - 4 . Newman , Eric P. ( 2008 ) . The
Early Paper Money of America ( 5 ed . ) . Krause Publications . ISBN 0 - 89689 - 326 - X . Schwartz , John ; Lindquist , Scott ( 2011 ) . Standard Guide to
Small - Size U.S. Paper Money -- 1928 to Date . Krause Publications . ISBN 978 - 1 - 4402 - 1703 - 6 . Retrieved 14 February 2014 . External links ( edit )
Media related to Money of the United States by denomination at Wikimedia Commons Large Denominations from the Bureau of Engraving and Printing U.S. Department
of the Treasury United States currency and coinage Topics Federal Reserve System Federal Reserve Note U.S. dollar U.S.
</item 2>

<item 3>
Large denominations of United States currency
See history of the United States dollar . Passive retirement ( edit ) Although they are still legal tender in the United States , high - denomination bills were
last printed on December 27 , 1945 , and officially discontinued on July 14 , 1969 , by the Federal Reserve System , due to ' lack of use ' . The $5,000 and
$10,000 effectively disappeared well before then . The Federal Reserve began taking high - denomination currency out of circulation ( destroying large bills
received by banks ) in 1969 . As of May 30 , 2009 , only 336 $10,000 bills were known to exist ; 342 remaining $5,000 bills ; and 165,372 remaining $1,000 bills
. Due to their rarity , collectors often pay considerably more than the face value of the bills to acquire them . Some are in museums in other parts of the
world . For the most part , these bills were used by banks and the Federal government for large financial transactions . This was especially true for gold
certificates from 1865 to 1934 .
</item 3>

<item 4>
Large denominations of United States currency
In popular culture ( edit ) Before they were ' retired ' , host Monty Hall regularly tempted or teased contestants on Let 's Make a Deal with a $1,000 bill
featuring Grover Cleveland . High - denomination ( HD ) banknote issuing data ( edit ) Key to high denomination banknote type abbreviations Abbr Type Size
Series dates High denomination series date Comments $500 $1,000 $5,000 $10,000 $100,000 LT Legal tender Large 1862 -- 1923 1862 1863 1869 1874 1875 1878 1880
1862 1863 1869 1878 1880 1878 1878 -- CITN Compound interest treasury note Exception 1863 -- 1864 1863 1864 1864 -- -- -- IBN Interest bearing note Exception
1861 -- 1865 1861 1863 1864 1865 1861 1863 1864 1865 1861 1863 1864 1865 -- -- SC Silver certificate Large 1878 -- 1923 1878 1880 1878 1880
</item 4>

<item 5>
Large denominations of United States currency
Large denominations of United States currency - wikipedia Large denominations of United States currency Jump to : navigation , search This article needs
additional citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and
removed . ( February 2016 ) ( Learn how and when to remove this template message ) Large denominations of United States currency greater than $100 were
circulated by the United States Treasury until 1969 . Since then , U.S. dollar banknotes have only been issued in seven denominations : $1 , $2 , $5 , $10 , $20
, $50 , and $100 . Contents ( hide ) 1 Overview and history 2 Public versus institutional use 3 Passive retirement 4 In popular culture 5 High - denomination (
HD ) banknote issuing data 6 High - denomination United States banknotes 7 See also 8 References 8.1 Footnotes 8.2 Notes 8.3 Bibliography 9 External links
Overview and history ( edit ) Large - denomination currency ( i.e.
</item 5>

<item 6>
Large denominations of United States currency
The display has since been dismantled and the bills sold to private collectors . Jump up ^ The following types of United States banknotes were not issued in
high denominations and are not included in the list below : Demand notes , Federal Reserve bank notes ( large or small size ) , legal tender ( small size ) ,
silver certificates ( small size ) , National Bank Notes ( small size ) The table sections are sorted by their appearance in the Friedberg reference book . Jump
up ^ Large size notes represent the earlier types or series of U.S. banknotes . Their `` average '' dimension is ​ 7 ⁄ × ​ 3 ⁄ inches ( 187 × 79 mm ) . Small
size notes ( described as such due to their size relative to the earlier large size notes ) are an `` average '' ​ 6 ⁄ × ​ 2 ⁄ ( 156 × 67 mm ) , the size of
modern U.S. currency . `` Each measurement is ± 0.08 inches ( 2 mm ) to account for margins and cutting '' .
</item 6>

<item 7>
Large denominations of United States currency
currency were issued in high - denomination notes across nearly 20 different series dates . The obverse of United States banknotes generally depict either
historical figures , allegorical figures symbolizing significant concepts ( e.g. , liberty , justice ) , or a combination of both . The reverse designs range
from abstract scroll - work with ornate denomination identifiers to reproductions of historical art works . Public versus institutional use ( edit ) Series 1934
gold certificates ( $100 , $1,000 , $10,000 and $100,000 ) were issued after the gold standard was repealed and gold was compulsorily confiscated by order of
President Franklin Roosevelt on March 9 , 1933 ( see United States Executive Order 6102 ) . Thus the series 1934 notes were used only for intragovernmental (
i.e. , Federal Reserve Bank ) transactions and were not issued to the public . This series was discontinued in 1940 . The series 1928 gold certificate reverse
was printed in black and green . See history of the United States dollar .
</item 7>

<item 8>
Large denominations of United States currency
1928 1934 -- GC Gold certificate Small 1928 -- 1934 1928 1928 1934 1928 1928 1934 1934 High - denomination United States banknotes ( edit ) The National
Numismatic Collection at the Smithsonian Institution contains ( among other things ) the Bureau of Engraving and Printing ( BEP ) certified proofs and the
Treasury Department collection of United States currency . Using a combination of proofs and issued notes , a nearly complete type set of high - denomination
currency was compiled . Notably missing are several types of Compound and Interest Bearing Notes . Printed during the early to mid-1860s on very thin paper ,
these high - denomination notes are virtually non-existent . Their issuance ( 1861 -- 65 ) pre-dates the BEP 's responsibility for U.S. Currency ( 1870s ) , so
it is fortunate that any proofs exist in the current archives . High denomination United States banknotes Value Type Series Fr . Image Portrait / engraving
Comments 000500 $500 LT 1862 -- 63 Fr.
</item 8>

<item 9>
Large denominations of United States currency
This was especially true for gold certificates from 1865 to 1934 . However , for the most part , the introduction of an electronic money system has made large -
scale cash transactions obsolete . When combined with concerns about counterfeiting and the use of cash in unlawful activities such as the illegal drug trade
and money laundering , it is unlikely that the U.S. government will re-issue large denomination currency in the near future , despite the amount of inflation
that has occurred since 1969 ( a $500 bill is now worth less , in real terms , than a $100 bill was worth in 1969 ) . According to the U.S. Department of
Treasury website , `` The present denominations of our currency in production are $1 , $2 , $5 , $10 , $20 , $50 and $100 . The purpose of the United States
currency system is to serve the needs of the public and these denominations meet that goal . Neither the Department of the Treasury nor the Federal Reserve
System has any plans to change the denominations in use today . ''
</item 9>

<item 10>
Large denominations of United States currency
379c Meade - George George Meade ( Charles Burt ) 2 known ( variety ) 3 known ( type ) 001000 $1,000 NBN 1865 -- 75 Fr. 465 Proof zScott - Winfield - Washington
- George Scott entering City of Mexico ( left ) ( Alfred Jones ) United States Capitol ( right ) ( James Smillie ) Washington resigning his commission ( rev ) (
Frederick Girsch ) Unknown 001000 $1,000 FRN 1918 Fr. 1133d Hamilton - Alexander Alexander Hamilton ( G.F.C. Smillie ) Eagle ( rev ) ( Marcus W. Baldwin )
001000 $1,000 GC 1863 Fr. 1166e Proof zEagle - shield - Justice - scales Eagle with shield or E Pluribus Unum ( Charles Skinner ) Justice with scales Unique
001000 $1,000 GC 1870 -- 75 Fr. 1166o Proof Hamilton - Alexander Alexander Hamilton ( Charles Burt ) Unique 001000 $1,000 GC 1882 Fr. 1218g Hamilton - Alexander
Alexander Hamilton ( G.F.C.
</item 10>
=== END ITEM id=d5e512aa0e3eee64 ===

=== ITEM id=c9b81864f5f630c9 ===
Gold sentences and facts:
Sentence 0: Only Mexico , Italy , France , Germany ( West Germany until shortly after the 1990 World Cup ) and Brazil have hosted the event on two occasions . Mexico City 's Estadio Azteca and Rio de Janeiro 's Maracanã are the only venues ever to have hosted two FIFA World Cup finals . Only the 2002 FIFA World Cup had more than one host , being split between Japan and South Korea .
  Fact 0.0: Only Mexico has hosted the event on two occasions.
  Fact 0.1: Only Italy has hosted the event on two occasions.
  Fact 0.2: Only France has hosted the event on two occasions.
  Fact 0.3: Only Germany has hosted the event on two occasions.
  Fact 0.4: Only Brazil has hosted the event on two occasions.
  Fact 0.5: Germany's hosting covers West Germany until shortly after the 1990 World Cup.
  Fact 0.6: Mexico City's Estadio Azteca is one of the only venues ever to have hosted two FIFA World Cup finals.
  Fact 0.7: Rio de Janeiro's Maracanã is one of the only venues ever to have hosted two FIFA World Cup finals.
  Fact 0.8: Estadio Azteca and the Maracanã are the only venues ever to have hosted two FIFA World Cup finals.
  Fact 0.9: Only the 2002 FIFA World Cup had more than one host.
  Fact 0.10: The 2002 FIFA World Cup was split between Japan and South Korea.

Retrieved items:
<item 1>
FIFA World Cup hosts
^ Jump up to : Joint bid by Libya and Tunisia . Host country performances ( edit ) Further information : National team appearances in the FIFA World Cup It is
widely considered that home advantage is common in the World Cup , with the host team usually performing above average . Of the eight countries that have won
the tournament , only Brazil and Spain were not champions at home , and both England and France had their only titles as hosts . Further , Sweden got to its
only final as hosts of the 1958 tournament , and only the Swedes and Brazilians finished as runner - up in home soil . The host country reached the semi-finals
thirteen times in twenty tournaments , and both Chile and South Korea had their only finishes on the top four at home . Only South Africa in 2010 managed to not
go past the group stage . # Performance Team Year Pld Win % GF GA GD GD / M GF / M Champion Uruguay 1930 0 0 100 % 15 12 3.8 Champion France 1998 7 6 0 93 % 15
13 1.9 2.
</item 1>

<item 2>
FIFA World Cup hosts
( PDF ) . PricewaterhouseCoopers . May 2010 . pp. 6 -- 7 . Retrieved 2 August 2014 . Jump up ^ Smallwood , John ( 11 June 2014 ) . `` Ultimate home field
advantage : Host nation luck '' . Philadelphia Daily News . Retrieved 2 August 2014 . External links ( edit ) FIFA World Cup Host Announcement Decision ( PDF )
FIFA World Cup hosts 1930 : Uruguay 1934 : Italy 1938 : France 1950 : Brazil 1954 : Switzerland 1958 : Sweden 1962 : Chile 1966 : England 1970 : Mexico 1974 :
West Germany 1978 : Argentina 1982 : Spain 1986 : Mexico 1990 : Italy 1994 : United States 1998 : France 2002 : South Korea & Japan 2006 : Germany 2010 : South
Africa 2014 : Brazil 2018 : Russia 2022 : Qatar FIFA World Cup Tournaments Uruguay 1930 Italy 1934 France 1938 Brazil 1950 Switzerland 1954 Sweden 1958 Chile
1962 England
</item 2>

<item 3>
FIFA World Cup hosts
33 1.3 12 Fourth place South Korea 2002 7 57 % 8 6 0.3 1.1 13 Fourth place Brazil 2014 7 57 % 11 14 − 3 − 0.4 1.57 14 Quarter - final Mexico 1986 5 0 80 % 6 0.8
15 Quarter - final Mexico 1970 63 % 6 0.5 1.5 16 Quarter - final Switzerland 1954 0 50 % 11 11 0 0 2.8 17 Quarter - final France 1938 0 50 % 0 0 18 Second round
Spain 1982 5 40 % 5 − 1 − 0.2 0.8 19 Round of 16 Japan 2002 63 % 5 0.5 1.25 20 Round of 16 United States 1994 38 % − 1 − 0.3 0.8 21 Group stage South Africa 50
% 5 − 2 − 0.
</item 3>

<item 4>
FIFA World Cup hosts
Both Argentina and Uruguay thus boycotted the tournament . The first tournament following World War II , held in Brazil in 1950 , had three teams withdraw due
to either financial problems or disagreements with the organization . To avoid any future boycotts or controversy , FIFA began a pattern of alternation between
the Americas and Europe , which continued until the 2002 FIFA World Cup in Asia . The system evolved so that the host country is now chosen in a vote by FIFA 's
Council ( former Executive Committee ) . This is done under an exhaustive ballot system . The decision is currently made roughly seven years in advance of the
tournament , though the hosts for the 2022 tournament were chosen at the same time as those for the 2018 tournament . Only Mexico , Italy , France , Germany (
West Germany until shortly after the 1990 World Cup ) and Brazil have hosted the event on two occasions . Mexico City 's Estadio Azteca and Rio de Janeiro 's
Maracanã are the only venues ever to have hosted two FIFA World Cup finals . Only the 2002 FIFA World Cup had more than one host , being split between Japan and
South Korea .
</item 4>

<item 5>
FIFA World Cup hosts
9 2.1 Champion England 1966 6 5 0 92 % 11 8 1.3 1.8 Champion Italy 1934 5 0 90 % 12 9 1.8 2.4 5 Champion West Germany 7 6 0 86 % 13 9 1.3 1.9 6 Champion
Argentina 1978 7 5 79 % 15 9 1.3 2.1 7 Runner - up Brazil 1950 6 75 % 22 6 16 2.7 3.7 8 Runner - up Sweden 1958 6 75 % 12 7 5 0.8 9 Third place Italy 1990 7 6 0
93 % 10 8 1.1 1.4 10 Third place Germany 2006 7 5 79 % 14 6 8 1.1 11 Third place Chile 1962 6 0 67 % 10 8 0.33 1.
</item 5>

<item 6>
FIFA World Cup hosts
FIFA World Cup bids 1930 Hungary Italy Netherlands Spain Sweden Uruguay 1934 Italy Sweden 1938 Argentina France Germany 1942 Brazil Germany 1946 None 1950
Brazil 1954 Switzerland 1958 Sweden 1962 Argentina Chile West Germany 1966 England West Germany Spain 1970 Argentina Mexico Italy Netherlands Spain West Germany
1978 Argentina Mexico 1982 Spain West Germany 1986 Canada Colombia Mexico United States 1990 Austria England France Greece Italy Soviet Union West Germany
Yugoslavia 1994 Brazil Morocco United States 1998 France Morocco Switzerland 2002 Japan -- South Korea Mexico 2006 Brazil England Germany Morocco South Africa
Egypt Libya -- Tunisia Morocco Nigeria South Africa 2014 Brazil Colombia 2018 Belgium -- Netherlands England Portugal -- Spain Russia 2022 Australia Japan Qatar
South Korea United States 2026 Canada -- Mexico -- United States Morocco 2030 Argentina -- Paraguay -- Uruguay England Notes : Bold entries are those chosen as
hosts Event cancelled due to World War II Colombia withdrew after being award event , which was re-awarded to Mexico Retrieved from ``
https://en.wikipedia.org/w/index.
</item 6>

<item 7>
FIFA World Cup hosts
1962 , 1966 , 1974 , 1982 , 2006 Argentina 5 1938 , 1962 , 1970 , 1978 , 2030 Mexico 1970 , 1978 , 1986 , 2002 , 2026 Morocco 1994 , 1998 , 2006 , 2010 , 2026
Spain 1930 , 1966 , 1974 , 1982 , 2018 Brazil 1950 , 1994 , 2006 , 2014 England 1966 , 1990 , 2006 , 2018 United States 1986 , 1994 , 2022 , 2026 Italy 1930 ,
1934 , 1990 Japan 1970 , 2002 , 2022 Canada 1986 , 2026 Chile 1962 , 2014 Colombia 1986 , 2014 France 1938 , 1998 Netherlands 1930 , 2018 Russia 1990 , 2018
South Africa 2006 , 2010 South Korea 2002 , 2022 Switzerland 1954 ,
</item 7>

<item 8>
FIFA World Cup hosts
1938 Brazil 1950 Switzerland 1954 Sweden 1958 Chile 1962 England 1966 Mexico 1970 West Germany 1974 Argentina 1978 Spain 1982 Mexico 1986 Italy 1990 United
States 1994 France 1998 South Korea / Japan 2002 Germany 2006 South Africa 2010 Brazil 2014 Russia 2018 Qatar 2022 2026 2030 2034 Qualification 1930 1934 1938
1950 1954 1958 1962 1966 1970 1978 1982 1986 1990 1994 1998 2002 2006 2014 2018 Finals 1930 1934 1938 1950 1954 1958 1962 1966 1970 1978 1982 1986 1990 1994
1998 2002 2006 2014 2018 Squads 1930 1934 1938 1950 1954 1958 1962 1966 1970 1978 1982 1986
</item 8>

<item 9>
FIFA World Cup hosts
Contents ( hide ) 1 List of hosts 2 1930 FIFA World Cup 3 1934 FIFA World Cup 4 1938 FIFA World Cup 5 Cancelled FIFA World Cups 1942 and 1946 6 1950 and 1954
FIFA World Cups 6.1 1949 bid 6.2 1951 bid 7 1958 FIFA World Cup 8 1962 FIFA World Cup 9 1966 FIFA World Cup 10 1970 FIFA World Cup 11 1974 , 1978 , 1982 FIFA
World Cups 11.1 1974 results 11.2 1978 results 11.3 1982 results 12 1986 FIFA World Cup 13 1990 FIFA World Cup 14 1994 FIFA World Cup 15 1998 FIFA World Cup 16
2002 FIFA World Cup 17 2006 FIFA World Cup 17.1 Controversy 18 2010 FIFA World Cup 18.1 Controversy 19 2014 FIFA World Cup 20 2018 and 2022 FIFA World Cups 20.1
Accusations of bribery 21
</item 9>

<item 10>
FIFA World Cup hosts
FIFA World Cup hosts - wikipedia FIFA World Cup hosts Jump to : navigation , search Map of FIFA World Cup final hosts , 1930 -- 2014 . Dark green : twice ;
middle green : once ; light green : planned until 2022 Sixteen countries have been FIFA World Cup hosts in the competition 's twenty tournaments since the
inaugural World Cup in 1930 . The organization at first awarded hosting to countries at meetings of FIFA 's congress . The choice of location was controversial
in the earliest tournaments , given the three - week boat journey between South America and Europe , the two centers of strength in football at the time . The
decision to hold the first cup in Uruguay , for example , led to only four European nations competing . The next two World Cups were both held in Europe . The
decision to hold the second of these , the 1938 FIFA World Cup , in France was controversial , as the South American countries had been led to understand that
the World Cup would rotate between the two continents . Both Argentina and Uruguay thus boycotted the tournament .
</item 10>
=== END ITEM id=c9b81864f5f630c9 ===

=== ITEM id=60daebbbf1127cc1 ===
Gold sentences and facts:
Sentence 0: Year Laureate Country Rationale 1901 Wilhelm Conrad Röntgen Germany `` in recognition of the extraordinary services he has rendered by the discovery of the remarkable rays subsequently named after him '' 1902 Hendrik Lorentz Netherlands `` in recognition of the extraordinary service they rendered by their researches into the influence of magnetism upon radiation phenomena '' Pieter Zeeman Netherlands 1903 Antoine Henri Becquerel France `` for his discovery of spontaneous radioactivity '' Pierre Curie France `` for their joint researches on the radiation phenomena discovered by Professor Henri Becquerel '' Maria Skłodowska - Curie Poland France 1904 Lord Rayleigh United Kingdom `` for his investigations of the densities of the most important gases and for his discovery of argon in connection with these studies '' 1905 Philipp Eduard Anton von Lenard Austria - Hungary Germany `` for his work on cathode rays '' 1906 Joseph John Thomson United Kingdom `` for his theoretical and experimental investigations on the conduction of electricity by gases '' 1907 Albert Abraham Michelson United States Poland `` for his optical precision instruments and the spectroscopic and metrological investigations carried out with their aid '' 1908 Gabriel Lippmann France `` for his method of reproducing colours photographically based on the phenomenon of interference '' 1909 Guglielmo Marconi Italy `` for their contributions to the development of wireless telegraphy '' Karl Ferdinand Braun Germany 1910 Johannes Diderik van der Waals Netherlands `` for his work on the equation of state for gases and liquids '' 1911 Wilhelm Wien Germany `` for his discoveries regarding the laws governing the radiation of heat '' 1912 Nils Gustaf Dalén Sweden `` for his invention of automatic valves designed to be used in combination with gas accumulators in lighthouses and buoys '' 1913 Heike Kamerlingh - Onnes Netherlands `` for his investigations on the properties of matter at low temperatures which led , inter alia , to the production of liquid helium '' 1914 Max von Laue Germany `` For his discovery of the diffraction of X-rays by crystals '' , an important step in the development of X-ray spectroscopy . 1915 William Henry Bragg United Kingdom `` For their services in the analysis of crystal structure by means of X-rays '' , an important step in the development of X-ray crystallography William Lawrence Bragg Australia United Kingdom 1916 Not awarded World War I 1917 Charles Glover Barkla United Kingdom `` For his discovery of the characteristic Röntgen radiation of the elements '' , another important step in the development of X-ray spectroscopy 1918 Max Planck Germany `` for the services he rendered to the advancement of physics by his discovery of energy quanta '' 1919 Johannes Stark Germany `` for his discovery of the Doppler effect in canal rays and the splitting of spectral lines in electric fields '' 1920 Charles Édouard Guillaume Switzerland `` for the service he has rendered to precision measurements in physics by his discovery of anomalies in nickel - steel alloys '' 1921 Albert Einstein Germany Switzerland `` for his services to theoretical physics , and especially for his discovery of the law of the photoelectric effect '' 1922 Niels Bohr Denmark `` for his services in the investigation of the structure of atoms and of the radiation emanating from them '' 1923 Robert Andrews Millikan United States `` for his work on the elementary charge of electricity and on the photoelectric effect '' 1924 Manne Siegbahn Sweden `` for his discoveries and research in the field of X-ray spectroscopy '' 1925 James Franck Germany `` for their discovery of the laws governing the impact of an electron upon an atom '' Gustav Hertz Germany 1926 Jean Baptiste Perrin France `` for his work on the discontinuous structure of matter , and especially for his discovery of sedimentation equilibrium '' 1927 Arthur Holly Compton United States `` for his discovery of the effect named after him '' Charles Thomson Rees Wilson United Kingdom `` for his method of making the paths of electrically charged particles visible by condensation of vapour '' 1928 Owen Willans Richardson United Kingdom `` for his work on the thermionic phenomenon and especially for the discovery of the law named after him '' 1929 Louis Victor Pierre Raymond , 7th Duc de Broglie France `` for his discovery of the wave nature of electrons '' 1930 Chandrasekhara Venkata Raman India `` for his work on the scattering of light and for the discovery of the effect named after him '' 1931 Not awarded 1932 Werner Heisenberg Germany `` for the creation of quantum mechanics , the application of which has , inter alia , led to the discovery of the allotropic forms of hydrogen '' 1933 Erwin Schrödinger Austria `` for the discovery of new productive forms of atomic theory '' Paul Dirac United Kingdom 1934 Not awarded 1935 James Chadwick United Kingdom `` for the discovery of the neutron '' 1936 Victor Francis Hess Austria `` for his discovery of cosmic radiation '' Carl David Anderson United States `` for his discovery of the positron '' 1937 Clinton Joseph Davisson United States `` for their experimental discovery of the diffraction of electrons by crystals '' George Paget Thomson United Kingdom 1938 Enrico Fermi Italy `` for his demonstrations of the existence of new radioactive elements produced by neutron irradiation , and for his related discovery of nuclear reactions brought about by slow neutrons '' 1939 Ernest Lawrence United States `` for the invention and development of the cyclotron and for results obtained with it , especially with regard to artificial radioactive elements '' 1940 Not awarded World War II 1941 Not awarded World War II 1942 Not awarded World War II 1943 Otto Stern United States `` for his contribution to the development of the molecular ray method and his discovery of the magnetic moment of the proton '' 1944 Isidor Isaac Rabi United States Poland `` for his resonance method for recording the magnetic properties of atomic nuclei '' 1945 Wolfgang Pauli Austria `` for the discovery of the Exclusion Principle , also called the Pauli principle '' 1946 Percy Williams Bridgman United States `` for the invention of an apparatus to produce extremely high pressures , and for the discoveries he made there within the field of high pressure physics '' 1947 Edward Victor Appleton United Kingdom `` for his investigations of the physics of the upper atmosphere especially for the discovery of the so - called Appleton layer '' 1948 Patrick Maynard Stuart Blackett United Kingdom `` for his development of the Wilson cloud chamber method , and his discoveries therewith in the fields of nuclear physics and cosmic radiation '' 1949 Hideki Yukawa Japan `` for his prediction of the existence of mesons on the basis of theoretical work on nuclear forces '' 1950 Cecil Frank Powell United Kingdom `` for his development of the photographic method of studying nuclear processes and his discoveries regarding mesons made with this method '' 1951 John Douglas Cockcroft United Kingdom `` for their pioneer work on the transmutation of atomic nuclei by artificially accelerated atomic particles '' Ernest Thomas Sinton Walton Ireland 1952 Felix Bloch Switzerland United States `` for their development of new methods for nuclear magnetic precision measurements and discoveries in connection therewith '' Edward Mills Purcell United States 1953 Frits Zernike Netherlands `` for his demonstration of the phase contrast method , especially for his invention of the phase contrast microscope '' 1954 Max Born West Germany `` for his fundamental research in quantum mechanics , especially for his statistical interpretation of the wavefunction '' Walther Bothe West Germany `` for the coincidence method and his discoveries made therewith '' 1955 Willis Eugene Lamb United States `` for his discoveries concerning the fine structure of the hydrogen spectrum '' Polykarp Kusch United States Germany `` for his precision determination of the magnetic moment of the electron '' 1956 John Bardeen United States `` for their researches on semiconductors and their discovery of the transistor effect '' Walter Houser Brattain United States William Bradford Shockley United States 1957 Tsung - Dao Lee Republic of China `` for their penetrating investigation of the so - called parity laws which has led to important discoveries regarding the elementary particles '' Chen Ning Yang Republic of China 1958 Pavel Alekseyevich Cherenkov Soviet Union `` for the discovery and the interpretation of the Cherenkov effect '' Ilya Frank Soviet Union Igor Yevgenyevich Tamm Soviet Union 1959 Emilio Gino Segrè Italy United States `` for their discovery of the antiproton '' Owen Chamberlain United States 1960 Donald Arthur Glaser United States `` for the invention of the bubble chamber '' 1961 Robert Hofstadter United States `` for his pioneering studies of electron scattering in atomic nuclei and for his thereby achieved discoveries concerning the structure of the nucleons '' Rudolf Ludwig Mössbauer West Germany `` for his researches concerning the resonance absorption of gamma radiation and his discovery in this connection of the effect which bears his name '' 1962 Lev Davidovich Landau Soviet Union `` for his pioneering theories for condensed matter , especially liquid helium '' 1963 Eugene Paul Wigner Hungary United States `` for his contributions to the theory of the atomic nucleus and the elementary particles , particularly through the discovery and application of fundamental symmetry principles '' Maria Goeppert - Mayer United States `` for their discoveries concerning nuclear shell structure '' J. Hans D. Jensen West Germany 1964 Nicolay Gennadiyevich Basov Soviet Union `` for fundamental work in the field of quantum electronics , which has led to the construction of oscillators and amplifiers based on the maser -- laser principle '' Alexander Prokhorov Soviet Union Charles Hard Townes United States 1965 Richard Phillips Feynman United States `` for their fundamental work in quantum electrodynamics ( QED ) , with deep - ploughing consequences for the physics of elementary particles '' Julian Schwinger United States Shin'ichirō Tomonaga Japan 1966 Alfred Kastler France `` for the discovery and development of optical methods for studying Hertzian resonances in atoms '' 1967 Hans Albrecht Bethe United States `` for his contributions to the theory of nuclear reactions , especially his discoveries concerning the energy production in stars '' 1968 Luis Walter Alvarez United States `` for his decisive contributions to elementary particle physics , in particular the discovery of a large number of resonance states , made possible through his development of the technique of using hydrogen bubble chamber and data analysis '' 1969 Murray Gell - Mann United States `` for his contributions and discoveries concerning the classification of elementary particles and their interactions '' 1970 Hannes Olof Gösta Alfvén Sweden `` for fundamental work and discoveries in magneto - hydrodynamics with fruitful applications in different parts of plasma physics '' Louis Néel France `` for fundamental work and discoveries concerning antiferromagnetism and ferrimagnetism which have led to important applications in solid state physics '' 1971 Dennis Gabor Hungary United Kingdom `` for his invention and development of the holographic method '' 1972 John Bardeen United States `` for their jointly developed theory of superconductivity , usually called the BCS - theory '' Leon Neil Cooper United States John Robert Schrieffer United States 1973 Leo Esaki Japan `` for their experimental discoveries regarding tunneling phenomena in semiconductors and superconductors , respectively '' Ivar Giaever United States Norway Brian David Josephson United Kingdom `` for his theoretical predictions of the properties of a supercurrent through a tunnel barrier , in particular those phenomena which are generally known as the Josephson effect '' Martin Ryle United Kingdom `` for their pioneering research in radio astrophysics : Ryle for his observations and inventions , in particular of the aperture synthesis technique , and Hewish for his decisive role in the discovery of pulsars '' Antony Hewish United Kingdom Aage Bohr Denmark `` for the discovery of the connection between collective motion and particle motion in atomic nuclei and the development of the theory of the structure of the atomic nucleus based on this connection '' Ben Roy Mottelson Denmark Leo James Rainwater United States 1976 Burton Richter United States `` for their pioneering work in the discovery of a heavy elementary particle of a new kind '' Samuel Chao Chung Ting United States 1977 Philip Warren Anderson United States `` for their fundamental theoretical investigations of the electronic structure of magnetic and disordered systems '' Nevill Francis Mott United Kingdom John Hasbrouck Van Vleck United States 1978 Pyotr Leonidovich Kapitsa Soviet Union `` for his basic inventions and discoveries in the area of low - temperature physics '' Arno Allan Penzias United States `` for their discovery of cosmic microwave background radiation '' Robert Woodrow Wilson United States 1979 Sheldon Lee Glashow United States `` for their contributions to the theory of the unified weak and electromagnetic interaction between elementary particles , including , inter alia , the prediction of the weak neutral current '' Abdus Salam Pakistan Steven Weinberg United States 1980 James Watson Cronin United States `` for the discovery of violations of fundamental symmetry principles in the decay of neutral K - mesons '' Val Logsdon Fitch United States 1981 Nicolaas Bloembergen Netherlands United States `` for their contribution to the development of laser spectroscopy '' Arthur Leonard Schawlow United States Kai Manne Börje Siegbahn Sweden `` for his contribution to the development of high - resolution electron spectroscopy '' 1982 Kenneth G. Wilson United States `` for his theory for critical phenomena in connection with phase transitions '' Subrahmanyan Chandrasekhar India United States `` for his theoretical studies of the physical processes of importance to the structure and evolution of the stars '' William Alfred Fowler United States `` for his theoretical and experimental studies of the nuclear reactions of importance in the formation of the chemical elements in the universe '' 1984 Carlo Rubbia Italy `` for their decisive contributions to the large project , which led to the discovery of the field particles W and Z , communicators of weak interaction '' Simon van der Meer Netherlands 1985 Klaus von Klitzing West Germany `` for the discovery of the quantized Hall effect '' 1986 Ernst Ruska West Germany `` for his fundamental work in electron optics , and for the design of the first electron microscope '' Gerd Binnig West Germany `` for their design of the scanning tunneling microscope '' Heinrich Rohrer Switzerland Johannes Georg Bednorz West Germany `` for their important break - through in the discovery of superconductivity in ceramic materials '' Karl Alexander Müller Switzerland 1988 Leon Max Lederman United States `` for the neutrino beam method and the demonstration of the doublet structure of the leptons through the discovery of the muon neutrino '' Melvin Schwartz United States Jack Steinberger United States 1989 Norman Foster Ramsey United States `` for the invention of the separated oscillatory fields method and its use in the hydrogen maser and other atomic clocks '' Hans Georg Dehmelt United States Germany `` for the development of the ion trap technique '' Wolfgang Paul West Germany 1990 Jerome I. Friedman United States `` for their pioneering investigations concerning deep inelastic scattering of electrons on protons and bound neutrons , which have been of essential importance for the development of the quark model in particle physics '' Henry Way Kendall United States Richard E. Taylor Canada 1991 Pierre - Gilles de Gennes France `` for discovering that methods developed for studying order phenomena in simple systems can be generalized to more complex forms of matter , in particular to liquid crystals and polymers '' 1992 Georges Charpak France Poland `` for his invention and development of particle detectors , in particular the multiwire proportional chamber '' 1993 Russell Alan Hulse United States `` for the discovery of a new type of pulsar , a discovery that has opened up new possibilities for the study of gravitation '' Joseph Hooton Taylor , Jr . United States 1994 Bertram Brockhouse Canada `` for the development of neutron spectroscopy '' and `` for pioneering contributions to the development of neutron scattering techniques for studies of condensed matter '' Clifford Glenwood Shull United States `` for the development of the neutron diffraction technique '' and `` for pioneering contributions to the development of neutron scattering techniques for studies of condensed matter '' 1995 Martin Lewis Perl United States `` for the discovery of the tau lepton '' and `` for pioneering experimental contributions to lepton physics '' Frederick Reines United States `` for the detection of the neutrino '' and `` for pioneering experimental contributions to lepton physics '' David Morris Lee United States `` for their discovery of superfluidity in helium - 3 '' Douglas D. Osheroff United States Robert Coleman Richardson United States 1997 Steven Chu United States `` for development of methods to cool and trap atoms with laser light . '' Claude Cohen - Tannoudji France William Daniel Phillips United States 1998 Robert B. Laughlin United States `` for their discovery of a new form of quantum fluid with fractionally charged excitations '' Horst Ludwig Störmer Germany Daniel Chee Tsui Republic of China United States 1999 Gerard ' t Hooft Netherlands `` for elucidating the quantum structure of electroweak interactions in physics '' Martinus J.G. Veltman Netherlands 2000 Zhores Ivanovich Alferov Russia `` for developing semiconductor heterostructures used in high - speed - and optoelectronics '' Herbert Kroemer Germany Jack St. Clair Kilby United States `` for his part in the invention of the integrated circuit '' 2001 Eric Allin Cornell United States `` for the achievement of Bose -- Einstein condensation in dilute gases of alkali atoms , and for early fundamental studies of the properties of the condensates '' Carl Edwin Wieman United States Wolfgang Ketterle Germany 2002 Raymond Davis , Jr . United States `` for pioneering contributions to astrophysics , in particular for the detection of cosmic neutrinos '' Masatoshi Koshiba Japan Riccardo Giacconi Italy United States `` for pioneering contributions to astrophysics , which have led to the discovery of cosmic X-ray sources '' 2003 Alexei Alexeyevich Abrikosov Russia United States `` for pioneering contributions to the theory of superconductors and superfluids '' Vitaly Lazarevich Ginzburg Russia Anthony James Leggett United Kingdom United States David J. Gross United States `` for the discovery of asymptotic freedom in the theory of the strong interaction '' Hugh David Politzer United States Frank Wilczek United States 2005 Roy J. Glauber United States `` for his contribution to the quantum theory of optical coherence '' John L. Hall United States `` for their contributions to the development of laser - based precision spectroscopy , including the optical frequency comb technique '' Theodor W. Hänsch Germany 2006 John C. Mather United States `` for their discovery of the blackbody form and anisotropy of the cosmic microwave background radiation '' George F. Smoot United States 2007 Albert Fert France `` for the discovery of giant magnetoresistance '' Peter Grünberg Germany 2008 Makoto Kobayashi Japan `` for the discovery of the origin of the broken symmetry which predicts the existence of at least three families of quarks in nature '' Toshihide Maskawa Japan Yoichiro Nambu Japan United States `` for the discovery of the mechanism of spontaneous broken symmetry in subatomic physics '' 2009 Charles K. Kao Hong Kong United Kingdom United States `` for groundbreaking achievements concerning the transmission of light in fibers for optical communication '' Willard S. Boyle Canada United States `` for the invention of an imaging semiconductor circuit -- the CCD sensor '' George E. Smith United States Andre Geim Russia United Kingdom Netherlands `` for groundbreaking experiments regarding the two - dimensional material graphene '' Konstantin Novoselov Russia United Kingdom 2011 Saul Perlmutter United States `` for the discovery of the accelerating expansion of the Universe through observations of distant supernovae '' Brian P. Schmidt Australia United States Adam G. Riess United States 2012 Serge Haroche France `` for ground - breaking experimental methods that enable measuring and manipulation of individual quantum systems . '' David J. Wineland United States 2013 François Englert Belgium `` for the theoretical discovery of a mechanism that contributes to our understanding of the origin of mass of subatomic particles , and which recently was confirmed through the discovery of the predicted fundamental particle , by the ATLAS and CMS experiments at CERN 's Large Hadron Collider '' Peter Higgs United Kingdom 2014 Isamu Akasaki Japan `` for the invention of efficient blue light - emitting diodes which has enabled bright and energy - saving white light sources '' Hiroshi Amano Japan Shuji Nakamura Japan United States 2015 Takaaki Kajita Japan `` for the discovery of neutrino oscillations , which shows that neutrinos have mass '' Arthur B. McDonald Canada 2016 David J. Thouless United Kingdom United States `` for theoretical discoveries of topological phase transitions and topological phases of matter '' F. Duncan M. Haldane United Kingdom United States John M. Kosterlitz United Kingdom United States 2017 Rainer Weiss Germany United States `` for decisive contributions to the LIGO detector and the observation of gravitational waves '' Kip Thorne United States Barry Barish United States
  Fact 0.0: 1901: Wilhelm Conrad Röntgen of Germany won the physics prize for the discovery of the remarkable rays later named after him
  Fact 0.1: 1902: Hendrik Lorentz of the Netherlands won the physics prize for research into the influence of magnetism upon radiation phenomena
  Fact 0.2: 1902: Pieter Zeeman of the Netherlands shared the physics prize with Hendrik Lorentz
  Fact 0.3: 1903: Antoine Henri Becquerel of France won the physics prize for the discovery of spontaneous radioactivity
  Fact 0.4: 1903: Pierre Curie of France shared the physics prize for joint research on the radiation phenomena discovered by Henri Becquerel
  Fact 0.5: 1903: Maria Skłodowska-Curie, listed as Poland and France, shared the physics prize for the same joint research on radiation phenomena
  Fact 0.6: 1904: Lord Rayleigh of the United Kingdom won the physics prize for investigating the densities of the most important gases
  Fact 0.7: 1904: Lord Rayleigh of the United Kingdom won the physics prize for discovering argon
  Fact 0.8: 1905: Philipp Eduard Anton von Lenard, listed as Austria-Hungary and Germany, won the physics prize for his work on cathode rays
  Fact 0.9: 1906: Joseph John Thomson of the United Kingdom won the physics prize for theoretical and experimental investigations of electricity conduction by gases
  Fact 0.10: 1907: Albert Abraham Michelson, listed as the United States and Poland, won the physics prize for optical precision instruments
  Fact 0.11: 1907: Albert Abraham Michelson won the physics prize for spectroscopic and metrological investigations carried out with those instruments
  Fact 0.12: 1908: Gabriel Lippmann of France won the physics prize for a method of reproducing colours photographically based on interference
  Fact 0.13: 1909: Guglielmo Marconi of Italy won the physics prize for contributions to the development of wireless telegraphy
  Fact 0.14: 1909: Karl Ferdinand Braun of Germany shared the physics prize for contributions to wireless telegraphy
  Fact 0.15: 1910: Johannes Diderik van der Waals of the Netherlands won the physics prize for work on the equation of state for gases and liquids
  Fact 0.16: 1911: Wilhelm Wien of Germany won the physics prize for discoveries regarding the laws governing the radiation of heat
  Fact 0.17: 1912: Nils Gustaf Dalén of Sweden won the physics prize for inventing automatic valves used with gas accumulators in lighthouses and buoys
  Fact 0.18: 1913: Heike Kamerlingh-Onnes of the Netherlands won the physics prize for investigations of matter at low temperatures, which led to the production of liquid helium
  Fact 0.19: 1914: Max von Laue of Germany won the physics prize for discovering the diffraction of X-rays by crystals
  Fact 0.20: 1915: William Henry Bragg of the United Kingdom shared the physics prize for analysing crystal structure by means of X-rays
  Fact 0.21: 1915: William Lawrence Bragg, listed as Australia and the United Kingdom, shared the physics prize for analysing crystal structure by means of X-rays
  Fact 0.22: 1916: No physics prize was awarded because of World War I
  Fact 0.23: 1917: Charles Glover Barkla of the United Kingdom won the physics prize for discovering the characteristic Röntgen radiation of the elements
  Fact 0.24: 1918: Max Planck of Germany won the physics prize for services to physics through his discovery of energy quanta
  Fact 0.25: 1919: Johannes Stark of Germany won the physics prize for discovering the Doppler effect in canal rays and the splitting of spectral lines in electric fields
  Fact 0.26: 1920: Charles Édouard Guillaume of Switzerland won the physics prize for precision measurements in physics through his discovery of anomalies in nickel-steel alloys
  Fact 0.27: 1921: Albert Einstein, listed as Germany and Switzerland, won the physics prize for services to theoretical physics
  Fact 0.28: 1921: Albert Einstein won the physics prize especially for discovering the law of the photoelectric effect
  Fact 0.29: 1922: Niels Bohr of Denmark won the physics prize for investigating the structure of atoms and the radiation emanating from them
  Fact 0.30: 1923: Robert Andrews Millikan of the United States won the physics prize for work on the elementary charge of electricity and on the photoelectric effect
  Fact 0.31: 1924: Manne Siegbahn of Sweden won the physics prize for discoveries and research in X-ray spectroscopy
  Fact 0.32: 1925: James Franck of Germany won the physics prize for discovering the laws governing the impact of an electron upon an atom
  Fact 0.33: 1925: Gustav Hertz of Germany shared the physics prize with James Franck
  Fact 0.34: 1926: Jean Baptiste Perrin of France won the physics prize for work on the discontinuous structure of matter, especially discovering sedimentation equilibrium
  Fact 0.35: 1927: Arthur Holly Compton of the United States won the physics prize for discovering the effect named after him
  Fact 0.36: 1927: Charles Thomson Rees Wilson of the United Kingdom shared the physics prize for a method of making charged particle paths visible by condensation of vapour
  Fact 0.37: 1928: Owen Willans Richardson of the United Kingdom won the physics prize for work on the thermionic phenomenon and discovering the law named after him
  Fact 0.38: 1929: Louis Victor Pierre Raymond, 7th Duc de Broglie, of France won the physics prize for discovering the wave nature of electrons
  Fact 0.39: 1930: Chandrasekhara Venkata Raman of India won the physics prize for work on the scattering of light and discovering the effect named after him
  Fact 0.40: 1931: No physics prize was awarded
  Fact 0.41: 1932: Werner Heisenberg of Germany won the physics prize for creating quantum mechanics, whose application led to the discovery of allotropic forms of hydrogen
  Fact 0.42: 1933: Erwin Schrödinger of Austria won the physics prize for discovering new productive forms of atomic theory
  Fact 0.43: 1933: Paul Dirac of the United Kingdom shared the physics prize for work on atomic theory
  Fact 0.44: 1934: No physics prize was awarded
  Fact 0.45: 1935: James Chadwick of the United Kingdom won the physics prize for discovering the neutron
  Fact 0.46: 1936: Victor Francis Hess of Austria won the physics prize for discovering cosmic radiation
  Fact 0.47: 1936: Carl David Anderson of the United States shared the physics prize for discovering the positron
  Fact 0.48: 1937: Clinton Joseph Davisson of the United States shared the physics prize for the experimental discovery of electron diffraction by crystals
  Fact 0.49: 1937: George Paget Thomson of the United Kingdom shared the physics prize for the experimental discovery of electron diffraction by crystals
  Fact 0.50: 1938: Enrico Fermi of Italy won the physics prize for demonstrating new radioactive elements produced by neutron irradiation
  Fact 0.51: 1938: Enrico Fermi won the physics prize for discovering nuclear reactions brought about by slow neutrons
  Fact 0.52: 1939: Ernest Lawrence of the United States won the physics prize for inventing and developing the cyclotron
  Fact 0.53: 1939: Ernest Lawrence won the physics prize for results obtained with the cyclotron, especially on artificial radioactive elements
  Fact 0.54: 1940: No physics prize was awarded because of World War II
  Fact 0.55: 1941: No physics prize was awarded because of World War II
  Fact 0.56: 1942: No physics prize was awarded because of World War II
  Fact 0.57: 1943: Otto Stern of the United States won the physics prize for developing the molecular ray method
  Fact 0.58: 1943: Otto Stern won the physics prize for discovering the magnetic moment of the proton
  Fact 0.59: 1944: Isidor Isaac Rabi, listed as the United States and Poland, won the physics prize for his resonance method for recording the magnetic properties of atomic nuclei
  Fact 0.60: 1945: Wolfgang Pauli of Austria won the physics prize for discovering the Exclusion Principle, also called the Pauli principle
  Fact 0.61: 1946: Percy Williams Bridgman of the United States won the physics prize for inventing an apparatus to produce extremely high pressures
  Fact 0.62: 1946: Percy Williams Bridgman won the physics prize for discoveries he made within high pressure physics
  Fact 0.63: 1947: Edward Victor Appleton of the United Kingdom won the physics prize for investigating the physics of the upper atmosphere
  Fact 0.64: 1947: Edward Victor Appleton won the physics prize especially for discovering the so-called Appleton layer
  Fact 0.65: 1948: Patrick Maynard Stuart Blackett of the United Kingdom won the physics prize for developing the Wilson cloud chamber method
  Fact 0.66: 1948: Patrick Maynard Stuart Blackett won the physics prize for discoveries with that method in nuclear physics and cosmic radiation
  Fact 0.67: 1949: Hideki Yukawa of Japan won the physics prize for predicting the existence of mesons from theoretical work on nuclear forces
  Fact 0.68: 1950: Cecil Frank Powell of the United Kingdom won the physics prize for developing the photographic method of studying nuclear processes
  Fact 0.69: 1950: Cecil Frank Powell won the physics prize for discoveries regarding mesons made with that method
  Fact 0.70: 1951: John Douglas Cockcroft of the United Kingdom shared the physics prize for pioneer work on transmuting atomic nuclei with artificially accelerated particles
  Fact 0.71: 1951: Ernest Thomas Sinton Walton of Ireland shared the physics prize for pioneer work on transmuting atomic nuclei with artificially accelerated particles
  Fact 0.72: 1952: Felix Bloch, listed as Switzerland and the United States, shared the physics prize for developing new methods for nuclear magnetic precision measurements
  Fact 0.73: 1952: Edward Mills Purcell of the United States shared the physics prize for developing new methods for nuclear magnetic precision measurements
  Fact 0.74: 1953: Frits Zernike of the Netherlands won the physics prize for demonstrating the phase contrast method
  Fact 0.75: 1953: Frits Zernike won the physics prize especially for inventing the phase contrast microscope
  Fact 0.76: 1954: Max Born of West Germany won the physics prize for fundamental research in quantum mechanics
  Fact 0.77: 1954: Max Born won the physics prize especially for his statistical interpretation of the wavefunction
  Fact 0.78: 1954: Walther Bothe of West Germany shared the physics prize for the coincidence method and his discoveries made with it
  Fact 0.79: 1955: Willis Eugene Lamb of the United States won the physics prize for discoveries concerning the fine structure of the hydrogen spectrum
  Fact 0.80: 1955: Polykarp Kusch, listed as the United States and Germany, shared the physics prize for precisely determining the magnetic moment of the electron
  Fact 0.81: 1956: John Bardeen of the United States shared the physics prize for researches on semiconductors and discovering the transistor effect
  Fact 0.82: 1956: Walter Houser Brattain of the United States shared the physics prize for researches on semiconductors and discovering the transistor effect
  Fact 0.83: 1956: William Bradford Shockley of the United States shared the physics prize for researches on semiconductors and discovering the transistor effect
  Fact 0.84: 1957: Tsung-Dao Lee of the Republic of China shared the physics prize for investigating the so-called parity laws
  Fact 0.85: 1957: Chen Ning Yang of the Republic of China shared the physics prize for investigating the so-called parity laws
  Fact 0.86: 1958: Pavel Alekseyevich Cherenkov of the Soviet Union shared the physics prize for discovering and interpreting the Cherenkov effect
  Fact 0.87: 1958: Ilya Frank of the Soviet Union shared the physics prize for discovering and interpreting the Cherenkov effect
  Fact 0.88: 1958: Igor Yevgenyevich Tamm of the Soviet Union shared the physics prize for discovering and interpreting the Cherenkov effect
  Fact 0.89: 1959: Emilio Gino Segrè, listed as Italy and the United States, shared the physics prize for discovering the antiproton
  Fact 0.90: 1959: Owen Chamberlain of the United States shared the physics prize for discovering the antiproton
  Fact 0.91: 1960: Donald Arthur Glaser of the United States won the physics prize for inventing the bubble chamber
  Fact 0.92: 1961: Robert Hofstadter of the United States shared the physics prize for pioneering studies of electron scattering in atomic nuclei
  Fact 0.93: 1961: Robert Hofstadter shared the physics prize for discoveries about the structure of the nucleons
  Fact 0.94: 1961: Rudolf Ludwig Mössbauer of West Germany shared the physics prize for researches on the resonance absorption of gamma radiation
  Fact 0.95: 1961: Rudolf Ludwig Mössbauer shared the physics prize for discovering the effect that bears his name
  Fact 0.96: 1962: Lev Davidovich Landau of the Soviet Union won the physics prize for pioneering theories of condensed matter, especially liquid helium
  Fact 0.97: 1963: Eugene Paul Wigner, listed as Hungary and the United States, shared the physics prize for contributions to the theory of the atomic nucleus and elementary particles
  Fact 0.98: 1963: Eugene Paul Wigner shared the physics prize for discovering and applying fundamental symmetry principles
  Fact 0.99: 1963: Maria Goeppert-Mayer of the United States shared the physics prize for discoveries concerning nuclear shell structure
  Fact 0.100: 1963: J. Hans D. Jensen of West Germany shared the physics prize for discoveries concerning nuclear shell structure
  Fact 0.101: 1964: Nicolay Gennadiyevich Basov of the Soviet Union shared the physics prize for fundamental work in quantum electronics on the maser-laser principle
  Fact 0.102: 1964: Alexander Prokhorov of the Soviet Union shared the physics prize for fundamental work in quantum electronics on the maser-laser principle
  Fact 0.103: 1964: Charles Hard Townes of the United States shared the physics prize for fundamental work in quantum electronics on the maser-laser principle
  Fact 0.104: 1965: Richard Phillips Feynman of the United States shared the physics prize for fundamental work in quantum electrodynamics
  Fact 0.105: 1965: Julian Schwinger of the United States shared the physics prize for fundamental work in quantum electrodynamics
  Fact 0.106: 1965: Shin'ichirō Tomonaga of Japan shared the physics prize for fundamental work in quantum electrodynamics
  Fact 0.107: 1966: Alfred Kastler of France won the physics prize for discovering and developing optical methods for studying Hertzian resonances in atoms
  Fact 0.108: 1967: Hans Albrecht Bethe of the United States won the physics prize for contributions to the theory of nuclear reactions
  Fact 0.109: 1967: Hans Albrecht Bethe won the physics prize especially for discoveries about energy production in stars
  Fact 0.110: 1968: Luis Walter Alvarez of the United States won the physics prize for decisive contributions to elementary particle physics
  Fact 0.111: 1968: Luis Walter Alvarez won the physics prize especially for discovering many resonance states using the hydrogen bubble chamber technique
  Fact 0.112: 1969: Murray Gell-Mann of the United States won the physics prize for contributions and discoveries concerning the classification of elementary particles and their interactions
  Fact 0.113: 1970: Hannes Olof Gösta Alfvén of Sweden shared the physics prize for fundamental work in magneto-hydrodynamics
  Fact 0.114: 1970: Louis Néel of France shared the physics prize for fundamental work and discoveries concerning antiferromagnetism and ferrimagnetism
  Fact 0.115: 1971: Dennis Gabor, listed as Hungary and the United Kingdom, won the physics prize for inventing and developing the holographic method
  Fact 0.116: 1972: John Bardeen of the United States shared the physics prize for the jointly developed BCS theory of superconductivity
  Fact 0.117: 1972: Leon Neil Cooper of the United States shared the physics prize for the jointly developed BCS theory of superconductivity
  Fact 0.118: 1972: John Robert Schrieffer of the United States shared the physics prize for the jointly developed BCS theory of superconductivity
  Fact 0.119: 1973: Leo Esaki of Japan shared the physics prize for experimental discoveries regarding tunneling phenomena in semiconductors
  Fact 0.120: 1973: Ivar Giaever, listed as the United States and Norway, shared the physics prize for experimental discoveries regarding tunneling phenomena in superconductors
  Fact 0.121: 1973: Brian David Josephson of the United Kingdom shared the physics prize for theoretical predictions of supercurrent properties through a tunnel barrier
  Fact 0.122: 1973: Brian David Josephson shared the physics prize for predicting the phenomena known as the Josephson effect
  Fact 0.123: 1974: Martin Ryle of the United Kingdom shared the physics prize for pioneering research in radio astrophysics
  Fact 0.124: 1974: Martin Ryle shared the physics prize for inventing the aperture synthesis technique
  Fact 0.125: 1974: Antony Hewish of the United Kingdom shared the physics prize for his decisive role in discovering pulsars
  Fact 0.126: 1975: Aage Bohr of Denmark shared the physics prize for discovering the connection between collective motion and particle motion in atomic nuclei
  Fact 0.127: 1975: Ben Roy Mottelson of Denmark shared the physics prize for developing the theory of atomic nucleus structure based on that connection
  Fact 0.128: 1975: Leo James Rainwater of the United States shared the physics prize for developing the theory of atomic nucleus structure based on that connection
  Fact 0.129: 1976: Burton Richter of the United States shared the physics prize for pioneering work in discovering a heavy elementary particle of a new kind
  Fact 0.130: 1976: Samuel Chao Chung Ting of the United States shared the physics prize for pioneering work in discovering a heavy elementary particle of a new kind
  Fact 0.131: 1977: Philip Warren Anderson of the United States shared the physics prize for theoretical investigations of the electronic structure of magnetic and disordered systems
  Fact 0.132: 1977: Nevill Francis Mott of the United Kingdom shared the physics prize for theoretical investigations of the electronic structure of magnetic and disordered systems
  Fact 0.133: 1977: John Hasbrouck Van Vleck of the United States shared the physics prize for theoretical investigations of the electronic structure of magnetic and disordered systems
  Fact 0.134: 1978: Pyotr Leonidovich Kapitsa of the Soviet Union shared the physics prize for basic inventions and discoveries in low-temperature physics
  Fact 0.135: 1978: Arno Allan Penzias of the United States shared the physics prize for discovering cosmic microwave background radiation
  Fact 0.136: 1978: Robert Woodrow Wilson of the United States shared the physics prize for discovering cosmic microwave background radiation
  Fact 0.137: 1979: Sheldon Lee Glashow of the United States shared the physics prize for contributions to the unified weak and electromagnetic interaction
  Fact 0.138: 1979: Sheldon Lee Glashow shared the physics prize for predicting the weak neutral current
  Fact 0.139: 1979: Abdus Salam of Pakistan shared the physics prize for contributions to the unified weak and electromagnetic interaction
  Fact 0.140: 1979: Steven Weinberg of the United States shared the physics prize for contributions to the unified weak and electromagnetic interaction
  Fact 0.141: 1980: James Watson Cronin of the United States shared the physics prize for discovering violations of fundamental symmetry principles in the decay of neutral K-mesons
  Fact 0.142: 1980: Val Logsdon Fitch of the United States shared the physics prize for discovering violations of fundamental symmetry principles in the decay of neutral K-mesons
  Fact 0.143: 1981: Nicolaas Bloembergen, listed as the Netherlands and the United States, shared the physics prize for contributions to laser spectroscopy
  Fact 0.144: 1981: Arthur Leonard Schawlow of the United States shared the physics prize for contributions to laser spectroscopy
  Fact 0.145: 1981: Kai Manne Börje Siegbahn of Sweden shared the physics prize for contributions to high-resolution electron spectroscopy
  Fact 0.146: 1982: Kenneth G. Wilson of the United States won the physics prize for his theory of critical phenomena in connection with phase transitions
  Fact 0.147: 1982: Subrahmanyan Chandrasekhar, listed as India and the United States, shared the physics prize for theoretical studies of physical processes important to the structure and evolution of stars
  Fact 0.148: 1982: William Alfred Fowler of the United States shared the physics prize for theoretical and experimental studies of nuclear reactions important in forming the chemical elements
  Fact 0.149: 1984: Carlo Rubbia of Italy shared the physics prize for decisive contributions to the large project that discovered the W and Z field particles
  Fact 0.150: 1984: Simon van der Meer of the Netherlands shared the physics prize for decisive contributions to the large project that discovered the W and Z field particles
  Fact 0.151: 1985: Klaus von Klitzing of West Germany won the physics prize for discovering the quantized Hall effect
  Fact 0.152: 1986: Ernst Ruska of West Germany shared the physics prize for fundamental work in electron optics
  Fact 0.153: 1986: Ernst Ruska shared the physics prize for designing the first electron microscope
  Fact 0.154: 1986: Gerd Binnig of West Germany shared the physics prize for designing the scanning tunneling microscope
  Fact 0.155: 1986: Heinrich Rohrer of Switzerland shared the physics prize for designing the scanning tunneling microscope
  Fact 0.156: 1986: Johannes Georg Bednorz of West Germany shared the physics prize for a breakthrough in discovering superconductivity in ceramic materials
  Fact 0.157: 1986: Karl Alexander Müller of Switzerland shared the physics prize for a breakthrough in discovering superconductivity in ceramic materials
  Fact 0.158: 1988: Leon Max Lederman of the United States shared the physics prize for the neutrino beam method
  Fact 0.159: 1988: Leon Max Lederman shared the physics prize for demonstrating the doublet structure of the leptons through discovering the muon neutrino
  Fact 0.160: 1988: Melvin Schwartz of the United States shared the physics prize for the neutrino beam method and the muon neutrino discovery
  Fact 0.161: 1988: Jack Steinberger of the United States shared the physics prize for the neutrino beam method and the muon neutrino discovery
  Fact 0.162: 1989: Norman Foster Ramsey of the United States shared the physics prize for inventing the separated oscillatory fields method
  Fact 0.163: 1989: Norman Foster Ramsey shared the physics prize for its use in the hydrogen maser and other atomic clocks
  Fact 0.164: 1989: Hans Georg Dehmelt, listed as the United States and Germany, shared the physics prize for developing the ion trap technique
  Fact 0.165: 1989: Wolfgang Paul of West Germany shared the physics prize for developing the ion trap technique
  Fact 0.166: 1990: Jerome I. Friedman of the United States shared the physics prize for pioneering investigations of deep inelastic scattering of electrons on protons and bound neutrons
  Fact 0.167: 1990: Henry Way Kendall of the United States shared the physics prize for pioneering investigations of deep inelastic scattering of electrons on protons and bound neutrons
  Fact 0.168: 1990: Richard E. Taylor of Canada shared the physics prize for pioneering investigations of deep inelastic scattering of electrons on protons and bound neutrons
  Fact 0.169: 1991: Pierre-Gilles de Gennes of France won the physics prize for discovering that methods for order phenomena in simple systems generalize to liquid crystals and polymers
  Fact 0.170: 1992: Georges Charpak, listed as France and Poland, won the physics prize for inventing and developing particle detectors
  Fact 0.171: 1992: Georges Charpak won the physics prize especially for the multiwire proportional chamber
  Fact 0.172: 1993: Russell Alan Hulse of the United States shared the physics prize for discovering a new type of pulsar
  Fact 0.173: 1993: Joseph Hooton Taylor Jr. of the United States shared the physics prize for discovering a new type of pulsar
  Fact 0.174: 1994: Bertram Brockhouse of Canada shared the physics prize for developing neutron spectroscopy
  Fact 0.175: 1994: Bertram Brockhouse shared the physics prize for pioneering contributions to neutron scattering techniques for studies of condensed matter
  Fact 0.176: 1994: Clifford Glenwood Shull of the United States shared the physics prize for developing the neutron diffraction technique
  Fact 0.177: 1994: Clifford Glenwood Shull shared the physics prize for pioneering contributions to neutron scattering techniques for studies of condensed matter
  Fact 0.178: 1995: Martin Lewis Perl of the United States shared the physics prize for discovering the tau lepton
  Fact 0.179: 1995: Martin Lewis Perl shared the physics prize for pioneering experimental contributions to lepton physics
  Fact 0.180: 1995: Frederick Reines of the United States shared the physics prize for detecting the neutrino
  Fact 0.181: 1995: Frederick Reines shared the physics prize for pioneering experimental contributions to lepton physics
  Fact 0.182: 1995: David Morris Lee of the United States shared the physics prize for discovering superfluidity in helium-3
  Fact 0.183: 1995: Douglas D. Osheroff of the United States shared the physics prize for discovering superfluidity in helium-3
  Fact 0.184: 1995: Robert Coleman Richardson of the United States shared the physics prize for discovering superfluidity in helium-3
  Fact 0.185: 1997: Steven Chu of the United States shared the physics prize for developing methods to cool and trap atoms with laser light
  Fact 0.186: 1997: Claude Cohen-Tannoudji of France shared the physics prize for developing methods to cool and trap atoms with laser light
  Fact 0.187: 1997: William Daniel Phillips of the United States shared the physics prize for developing methods to cool and trap atoms with laser light
  Fact 0.188: 1998: Robert B. Laughlin of the United States shared the physics prize for discovering a new form of quantum fluid with fractionally charged excitations
  Fact 0.189: 1998: Horst Ludwig Störmer of Germany shared the physics prize for discovering a new form of quantum fluid with fractionally charged excitations
  Fact 0.190: 1998: Daniel Chee Tsui, listed as the Republic of China and the United States, shared the physics prize for discovering a new form of quantum fluid with fractionally charged excitations
  Fact 0.191: 1999: Gerard 't Hooft of the Netherlands shared the physics prize for elucidating the quantum structure of electroweak interactions
  Fact 0.192: 1999: Martinus J.G. Veltman of the Netherlands shared the physics prize for elucidating the quantum structure of electroweak interactions
  Fact 0.193: 2000: Zhores Ivanovich Alferov of Russia shared the physics prize for developing semiconductor heterostructures used in high-speed and optoelectronics
  Fact 0.194: 2000: Herbert Kroemer of Germany shared the physics prize for developing semiconductor heterostructures used in high-speed and optoelectronics
  Fact 0.195: 2000: Jack St. Clair Kilby of the United States shared the physics prize for his part in the invention of the integrated circuit
  Fact 0.196: 2001: Eric Allin Cornell of the United States shared the physics prize for achieving Bose-Einstein condensation in dilute gases of alkali atoms
  Fact 0.197: 2001: Carl Edwin Wieman of the United States shared the physics prize for early fundamental studies of the properties of the condensates
  Fact 0.198: 2001: Wolfgang Ketterle of Germany shared the physics prize for achieving Bose-Einstein condensation in dilute gases of alkali atoms
  Fact 0.199: 2002: Raymond Davis Jr. of the United States shared the physics prize for pioneering contributions to astrophysics through the detection of cosmic neutrinos
  Fact 0.200: 2002: Masatoshi Koshiba of Japan shared the physics prize for pioneering contributions to astrophysics through the detection of cosmic neutrinos
  Fact 0.201: 2002: Riccardo Giacconi, listed as Italy and the United States, shared the physics prize for pioneering contributions to astrophysics that led to the discovery of cosmic X-ray sources
  Fact 0.202: 2003: Alexei Alexeyevich Abrikosov, listed as Russia and the United States, shared the physics prize for pioneering contributions to the theory of superconductors and superfluids
  Fact 0.203: 2003: Vitaly Lazarevich Ginzburg of Russia shared the physics prize for pioneering contributions to the theory of superconductors and superfluids
  Fact 0.204: 2003: Anthony James Leggett, listed as the United Kingdom and the United States, shared the physics prize for pioneering contributions to the theory of superconductors and superfluids
  Fact 0.205: 2003: David J. Gross of the United States shared the physics prize for discovering asymptotic freedom in the theory of the strong interaction
  Fact 0.206: 2003: Hugh David Politzer of the United States shared the physics prize for discovering asymptotic freedom in the theory of the strong interaction
  Fact 0.207: 2003: Frank Wilczek of the United States shared the physics prize for discovering asymptotic freedom in the theory of the strong interaction
  Fact 0.208: 2005: Roy J. Glauber of the United States won the physics prize for his contribution to the quantum theory of optical coherence
  Fact 0.209: 2005: John L. Hall of the United States shared the physics prize for contributions to laser-based precision spectroscopy, including the optical frequency comb technique
  Fact 0.210: 2005: Theodor W. Hänsch of Germany shared the physics prize for contributions to laser-based precision spectroscopy, including the optical frequency comb technique
  Fact 0.211: 2006: John C. Mather of the United States shared the physics prize for discovering the blackbody form of the cosmic microwave background radiation
  Fact 0.212: 2006: John C. Mather shared the physics prize for discovering the anisotropy of the cosmic microwave background radiation
  Fact 0.213: 2006: George F. Smoot of the United States shared the physics prize for discovering the blackbody form of the cosmic microwave background radiation
  Fact 0.214: 2006: George F. Smoot shared the physics prize for discovering the anisotropy of the cosmic microwave background radiation
  Fact 0.215: 2007: Albert Fert of France shared the physics prize for discovering giant magnetoresistance
  Fact 0.216: 2007: Peter Grünberg of Germany shared the physics prize for discovering giant magnetoresistance
  Fact 0.217: 2008: Makoto Kobayashi of Japan shared the physics prize for discovering the origin of the broken symmetry that predicts at least three families of quarks in nature
  Fact 0.218: 2008: Toshihide Maskawa of Japan shared the physics prize for discovering the origin of the broken symmetry that predicts at least three families of quarks in nature
  Fact 0.219: 2008: Yoichiro Nambu, listed as Japan and the United States, shared the physics prize for discovering the mechanism of spontaneous broken symmetry in subatomic physics
  Fact 0.220: 2009: Charles K. Kao, listed as Hong Kong, the United Kingdom and the United States, shared the physics prize for groundbreaking achievements concerning light transmission in fibers for optical communication
  Fact 0.221: 2009: Willard S. Boyle, listed as Canada and the United States, shared the physics prize for inventing an imaging semiconductor circuit, the CCD sensor
  Fact 0.222: 2009: George E. Smith of the United States shared the physics prize for inventing an imaging semiconductor circuit, the CCD sensor
  Fact 0.223: 2010: Andre Geim, listed as Russia, the United Kingdom and the Netherlands, shared the physics prize for groundbreaking experiments regarding the two-dimensional material graphene
  Fact 0.224: 2010: Konstantin Novoselov, listed as Russia and the United Kingdom, shared the physics prize for groundbreaking experiments regarding the two-dimensional material graphene
  Fact 0.225: 2011: Saul Perlmutter of the United States shared the physics prize for discovering the accelerating expansion of the Universe through observations of distant supernovae
  Fact 0.226: 2011: Brian P. Schmidt, listed as Australia and the United States, shared the physics prize for discovering the accelerating expansion of the Universe through observations of distant supernovae
  Fact 0.227: 2011: Adam G. Riess of the United States shared the physics prize for discovering the accelerating expansion of the Universe through observations of distant supernovae
  Fact 0.228: 2012: Serge Haroche of France shared the physics prize for ground-breaking experimental methods that enable measuring and manipulating individual quantum systems
  Fact 0.229: 2012: David J. Wineland of the United States shared the physics prize for ground-breaking experimental methods that enable measuring and manipulating individual quantum systems
  Fact 0.230: 2013: François Englert of Belgium shared the physics prize for the theoretical discovery of a mechanism explaining the origin of mass of subatomic particles
  Fact 0.231: 2013: Peter Higgs of the United Kingdom shared the physics prize for the theoretical discovery of a mechanism explaining the origin of mass of subatomic particles
  Fact 0.232: 2014: Isamu Akasaki of Japan shared the physics prize for inventing efficient blue light-emitting diodes
  Fact 0.233: 2014: Hiroshi Amano of Japan shared the physics prize for inventing efficient blue light-emitting diodes
  Fact 0.234: 2014: Shuji Nakamura, listed as Japan and the United States, shared the physics prize for inventing efficient blue light-emitting diodes
  Fact 0.235: 2015: Takaaki Kajita of Japan shared the physics prize for discovering neutrino oscillations, which shows that neutrinos have mass
  Fact 0.236: 2015: Arthur B. McDonald of Canada shared the physics prize for discovering neutrino oscillations, which shows that neutrinos have mass
  Fact 0.237: 2016: David J. Thouless, listed as the United Kingdom and the United States, shared the physics prize for theoretical discoveries of topological phase transitions and topological phases of matter
  Fact 0.238: 2016: F. Duncan M. Haldane, listed as the United Kingdom and the United States, shared the physics prize for theoretical discoveries of topological phase transitions and topological phases of matter
  Fact 0.239: 2016: John M. Kosterlitz, listed as the United Kingdom and the United States, shared the physics prize for theoretical discoveries of topological phase transitions and topological phases of matter
  Fact 0.240: 2017: Rainer Weiss, listed as Germany and the United States, shared the physics prize for decisive contributions to the LIGO detector and the observation of gravitational waves
  Fact 0.241: 2017: Kip Thorne of the United States shared the physics prize for decisive contributions to the LIGO detector and the observation of gravitational waves
  Fact 0.242: 2017: Barry Barish of the United States shared the physics prize for decisive contributions to the LIGO detector and the observation of gravitational waves
Sentence 1: The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen , of Germany , who received 150,782 SEK , which is equal to 7,731,004 SEK in December 2007 . John Bardeen is the only laureate to win the prize twice -- in 1956 and 1972 . Maria Skłodowska - Curie also won two Nobel Prizes , for physics in 1903 and chemistry in 1911 . William Lawrence Bragg was , until October 2014 , the youngest ever Nobel laureate ; he won the prize in 1915 at the age of 25 . Two women have won the prize : Curie and Maria Goeppert - Mayer ( 1963 ) . As of 2017 , the prize has been awarded to 206 individuals . There have been six years in which the Nobel Prize in Physics was not awarded ( 1916 , 1931 , 1934 , 1940 -- 1942 ) .
  Fact 1.0: The first Nobel Prize in Physics was awarded in 1901
  Fact 1.1: The first Nobel Prize in Physics went to Wilhelm Conrad Röntgen of Germany
  Fact 1.2: Wilhelm Conrad Röntgen received 150,782 SEK
  Fact 1.3: 150,782 SEK is equal to 7,731,004 SEK in December 2007
  Fact 1.4: John Bardeen is the only laureate to win the prize twice
  Fact 1.5: John Bardeen won the prize in 1956 and in 1972
  Fact 1.6: Maria Skłodowska-Curie also won two Nobel Prizes
  Fact 1.7: Maria Skłodowska-Curie won the physics prize in 1903
  Fact 1.8: Maria Skłodowska-Curie won the chemistry prize in 1911
  Fact 1.9: William Lawrence Bragg was, until October 2014, the youngest ever Nobel laureate
  Fact 1.10: William Lawrence Bragg won the prize in 1915 at the age of 25
  Fact 1.11: Two women have won the prize: Curie and Maria Goeppert-Mayer
  Fact 1.12: Maria Goeppert-Mayer won the prize in 1963
  Fact 1.13: As of 2017, the prize has been awarded to 206 individuals
  Fact 1.14: There have been six years in which the Nobel Prize in Physics was not awarded: 1916, 1931, 1934, 1940, 1941 and 1942

Retrieved items:
<item 1>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1925 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1926 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . ^ Jump up to : `` The Nobel Prize in Physics 1927 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The
Nobel Prize in Physics 1928 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1929 '' . Nobel Foundation . Retrieved
2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1930 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1932 ''
. Nobel Foundation . Retrieved 2008 - 10 - 09 .
</item 1>

<item 2>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1918 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1919 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1920 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel
Prize in Physics 1921 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1922 '' . Nobel Foundation . Retrieved 2008 -
10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1923 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1924 '' .
Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1925 '' .
</item 2>

<item 3>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1943 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1944 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1945 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel
Prize in Physics 1946 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1947 '' . Nobel Foundation . Retrieved 2008 -
10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1948 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1949 '' .
Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1950 '' .
</item 3>

<item 4>
List of Nobel laureates in Physics
Smoot United States 2007 Albert Fert France `` for the discovery of giant magnetoresistance '' Peter Grünberg Germany 2008 Makoto Kobayashi Japan `` for the
discovery of the origin of the broken symmetry which predicts the existence of at least three families of quarks in nature '' Toshihide Maskawa Japan Yoichiro
Nambu Japan United States `` for the discovery of the mechanism of spontaneous broken symmetry in subatomic physics '' 2009 Charles K. Kao Hong Kong United
Kingdom United States `` for groundbreaking achievements concerning the transmission of light in fibers for optical communication '' Willard S. Boyle Canada
United States `` for the invention of an imaging semiconductor circuit -- the CCD sensor '' George E. Smith United States Andre Geim Russia United Kingdom
Netherlands `` for groundbreaking experiments regarding the two - dimensional material graphene '' Konstantin Novoselov Russia United Kingdom 2011 Saul
Perlmutter United States `` for the discovery of the accelerating expansion of the Universe through observations of distant supernovae '' Brian P. Schmidt
Australia United States Adam G. Riess United States 2012 Serge Haroche France `` for ground - breaking experimental methods that enable measuring and
manipulation of individual quantum systems . ''
</item 4>

<item 5>
List of Nobel laureates in Physics
As of 2017 , the prize has been awarded to 206 individuals . There have been six years in which the Nobel Prize in Physics was not awarded ( 1916 , 1931 , 1934
, 1940 -- 1942 ) . Laureates ( edit ) Year Laureate Country Rationale 1901 Wilhelm Conrad Röntgen Germany `` in recognition of the extraordinary services he has
rendered by the discovery of the remarkable rays subsequently named after him '' 1902 Hendrik Lorentz Netherlands `` in recognition of the extraordinary service
they rendered by their researches into the influence of magnetism upon radiation phenomena '' Pieter Zeeman Netherlands 1903 Antoine Henri Becquerel France ``
for his discovery of spontaneous radioactivity '' Pierre Curie France `` for their joint researches on the radiation phenomena discovered by Professor Henri
Becquerel '' Maria Skłodowska - Curie Poland France 1904 Lord Rayleigh United Kingdom `` for his investigations of the densities of the most important gases and
for his discovery of argon in connection with these studies '' 1905 Philipp Eduard Anton von Lenard Austria - Hungary Germany `` for his work on cathode rays ''
1906 Joseph
</item 5>

<item 6>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1910 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1911 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1912 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel
Prize in Physics 1913 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1914 '' . Nobel Foundation . Retrieved 2008 -
10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1915 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1917 '' .
Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1918 '' .
</item 6>

<item 7>
List of Nobel laureates in Physics
Contents ( hide ) 1 Statistics 2 Laureates 3 See also 4 References 4.1 Notes 4.2 Citations 4.3 Sources 5 External links Statistics ( edit ) The first Nobel
Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen , of Germany , who received 150,782 SEK , which is equal to 7,731,004 SEK in December 2007 . John
Bardeen is the only laureate to win the prize twice -- in 1956 and 1972 . Maria Skłodowska - Curie also won two Nobel Prizes , for physics in 1903 and chemistry
in 1911 . William Lawrence Bragg was , until October 2014 , the youngest ever Nobel laureate ; he won the prize in 1915 at the age of 25 . Two women have won
the prize : Curie and Maria Goeppert - Mayer ( 1963 ) . As of 2017 , the prize has been awarded to 206 individuals .
</item 7>

<item 8>
List of Nobel laureates in Physics
Yang / T.D. Lee 1958 Cherenkov / Frank / Tamm 1959 Segrè / Chamberlain 1960 Glaser 1961 Hofstadter / Mössbauer 1962 Landau 1963 Wigner / Goeppert - Mayer /
Jensen 1964 Townes / Basov / Prokhorov 1965 Tomonaga / Schwinger / Feynman 1966 Kastler 1967 Bethe 1968 Alvarez 1969 Gell - Mann 1970 Alfvén / Néel 1971 Gabor
1972 Bardeen / Cooper / Schrieffer 1973 Esaki / Giaever / Josephson 1974 Ryle / Hewish 1975 A. Bohr / Mottelson / Rainwater 1976 -- 2000 1976 Richter / Ting
1977 P.W. Anderson / Mott / Van Vleck 1978 Kapitsa / Penzias / R. Wilson 1979 Glashow / Salam / Weinberg 1980 Cronin / Fitch 1981 Bloembergen / Schawlow / K.
</item 8>

<item 9>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1950 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1951 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1952 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel
Prize in Physics 1953 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . ^ Jump up to : `` The Nobel Prize in Physics 1954 '' . Nobel Foundation . Retrieved
2008 - 10 - 09 . ^ Jump up to : `` The Nobel Prize in Physics 1955 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics
1956 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 .
</item 9>

<item 10>
List of Nobel laureates in Physics
Claude Cohen - Tannoudji France William Daniel Phillips United States 1998 Robert B. Laughlin United States `` for their discovery of a new form of quantum
fluid with fractionally charged excitations '' Horst Ludwig Störmer Germany Daniel Chee Tsui Republic of China United States 1999 Gerard ' t Hooft Netherlands
`` for elucidating the quantum structure of electroweak interactions in physics '' Martinus J.G. Veltman Netherlands 2000 Zhores Ivanovich Alferov Russia `` for
developing semiconductor heterostructures used in high - speed - and optoelectronics '' Herbert Kroemer Germany Jack St. Clair Kilby United States `` for his
part in the invention of the integrated circuit '' 2001 Eric Allin Cornell United States `` for the achievement of Bose -- Einstein condensation in dilute gases
of alkali atoms , and for early fundamental studies of the properties of the condensates '' Carl Edwin Wieman United States Wolfgang Ketterle Germany 2002
Raymond Davis , Jr . United States `` for pioneering contributions to astrophysics ,
</item 10>
=== END ITEM id=60daebbbf1127cc1 ===

=== ITEM id=76a164a70041a2bc ===
Gold sentences and facts:
Sentence 0: John James McCoy is a fictional character in the television drama Law & Order . He was created by Michael S. Chernuchin and portrayed by Sam Waterston from 1994 until the end of the series in 2010 . He is the second - longest tenured character on the show ( 16 seasons ) , after Lt. Anita Van Buren ( 17 seasons ; portrayed by S. Epatha Merkerson ) . He appeared in 368 episodes of Law & Order , four episodes of Law & Order : Special Victims Unit , two episodes of Law & Order : Trial by Jury , two episodes of Homicide : Life on the Street , and the made - for - TV movie Exiled .
  Fact 0.0: John James McCoy is a fictional character in the television drama Law & Order.
  Fact 0.1: John James McCoy was created by Michael S. Chernuchin.
  Fact 0.2: John James McCoy was portrayed by Sam Waterston from 1994 until the end of the series in 2010.
  Fact 0.3: John James McCoy is the second-longest tenured character on the show, with 16 seasons.
  Fact 0.4: John James McCoy's tenure ranks just after Lt. Anita Van Buren, who had 17 seasons.
  Fact 0.5: Lt. Anita Van Buren was portrayed by S. Epatha Merkerson.
  Fact 0.6: John James McCoy appeared in 368 episodes of Law & Order.
  Fact 0.7: John James McCoy appeared in four episodes of Law & Order: Special Victims Unit.
  Fact 0.8: John James McCoy appeared in two episodes of Law & Order: Trial by Jury.
  Fact 0.9: John James McCoy appeared in two episodes of Homicide: Life on the Street.
  Fact 0.10: John James McCoy appeared in the made-for-TV movie Exiled.
Sentence 1: Jack McCoy brings 24 years of experience with him as he is appointed executive assistant district attorney by Adam Schiff ( Steven Hill ) in the season - five premiere episode `` Second Opinion '' . He quickly establishes himself as a more unconventional , ruthless litigator than his predecessor , Ben Stone ( Michael Moriarty ) . He often bends -- and sometimes breaks -- trial rules to get convictions , finds tenuous rationales for charging defendants with crimes when the original charges fail to stick , and charges innocent people to frighten them into testifying against others . McCoy is found in contempt of court 80 times for such behavior , and his tactics occasionally incur negative publicity for the DA 's office . His underlying motivation , however , is not , he maintains , corruption , but a sincere desire to see justice done . To that end , McCoy has gone after defendants accused of perverting the justice system to arrange wrongful convictions with just as much determination as his more mundane cases . Such aggressive actions in the courts have earned him the nickname `` Hang ' em High McCoy '' . He has subsequently developed a reputation with both colleagues and rival attorneys , once being referred to as `` the top of the legal food chain '' by a rival attorney during a trial .
  Fact 1.0: Jack McCoy brings 24 years of experience with him.
  Fact 1.1: Jack McCoy is appointed executive assistant district attorney by Adam Schiff.
  Fact 1.2: Adam Schiff is played by Steven Hill.
  Fact 1.3: Jack McCoy is appointed in the season-five premiere episode 'Second Opinion'.
  Fact 1.4: Jack McCoy quickly establishes himself as a more unconventional, ruthless litigator than his predecessor.
  Fact 1.5: Jack McCoy's predecessor is Ben Stone.
  Fact 1.6: Ben Stone is played by Michael Moriarty.
  Fact 1.7: Jack McCoy often bends trial rules to get convictions.
  Fact 1.8: Jack McCoy sometimes breaks trial rules to get convictions.
  Fact 1.9: Jack McCoy finds tenuous rationales for charging defendants with crimes when the original charges fail to stick.
  Fact 1.10: Jack McCoy charges innocent people to frighten them into testifying against others.
  Fact 1.11: Jack McCoy is found in contempt of court 80 times for such behavior.
  Fact 1.12: Jack McCoy's tactics occasionally incur negative publicity for the DA's office.
  Fact 1.13: Jack McCoy's underlying motivation is not corruption, he maintains.
  Fact 1.14: Jack McCoy's underlying motivation is a sincere desire to see justice done.
  Fact 1.15: Jack McCoy has gone after defendants accused of perverting the justice system to arrange wrongful convictions.
  Fact 1.16: Jack McCoy pursues those defendants with just as much determination as his more mundane cases.
  Fact 1.17: Jack McCoy's aggressive actions in the courts have earned him the nickname 'Hang 'em High McCoy'.
  Fact 1.18: Jack McCoy has developed a reputation with both colleagues and rival attorneys.
  Fact 1.19: Jack McCoy was once referred to as 'the top of the legal food chain' by a rival attorney during a trial.
Sentence 2: `` Entitled '' ( onscreen ) : McCoy makes his first appearance on the series . McCoy , working with his assistant , Abbie Carmichael ( Angie Harmon ) , assists the Special Victims Unit in solving a cold case , that Briscoe handled years ago with his former partner , Detective Mike Logan ( Chris Noth ) .
  Fact 2.0: The episode 'Entitled' (onscreen) is where McCoy makes his first appearance on the series.
  Fact 2.1: McCoy works with his assistant, Abbie Carmichael.
  Fact 2.2: Abbie Carmichael is played by Angie Harmon.
  Fact 2.3: McCoy assists the Special Victims Unit in solving a cold case.
  Fact 2.4: Briscoe handled the cold case years ago.
  Fact 2.5: Briscoe handled the cold case with his former partner, Detective Mike Logan.
  Fact 2.6: Detective Mike Logan is played by Chris Noth.

Retrieved items:
<item 1>
Jack McCoy
Reception ( edit ) Entertainment Weekly television critic Ken Tucker has praised Law & Order 's creator Dick Wolf for putting McCoy at the center of `` some of
the best episodes of the immortal series ' 19th season . '' Tucker elaborates how the character , riding `` herd over a couple of stubborn young bucks --
assistant DAs Mike Cutter ( Linus Roache ) and Connie Rubirosa ( Alana de la Garza ) -- McCoy argues , bellows orders , and croaks with outrage when his charges
disobey his legal advice . ''
</item 1>

<item 2>
Jack McCoy
Law & Order . Season 6 . Episode 23 . May 22 , 1996 . NBC . ^ Jump up to : `` In Vino Veritas '' . Law & Order . Season 17 . Episode 17 . November 3 , 2006 .
NBC . Jump up ^ `` Wannabe '' . Law & Order . Season 5 . Episode 16 . March 15 , 1995 . NBC . Jump up ^ `` Patient Zero '' . Law & Order . Season 14 . Episode 3
. October 8 , 2003 . NBC . Jump up ^ `` Fame '' . Law & Order . Season 17 . Episode 1 . September 22 , 2006 . NBC . Jump up ^ `` Excalibur '' . Law & Order .
Season 18 . Episode 18 . May 21 , 2008 . NBC . Jump up ^ `` Loco Parentis '' . Law & Order . Season 10 . Episode 10 . January 5 , 2000 . NBC .
</item 2>

<item 3>
Jack McCoy
District attorney 's office timeline ( edit ) Time period Executive assistant district attorney ( EADA ) Assistant district attorney ( ADA ) District attorney (
DA ) 1994 -- 1996 Jack McCoy Claire Kincaid Adam Schiff 1996 -- 1998 Jamie Ross 1998 -- 2000 Abbie Carmichael 2000 -- 2001 Nora Lewin 2001 -- 2002 Serena
Southerlyn 2002 -- 2005 Arthur Branch 2005 -- 2006 Alexandra Borgia 2006 -- 2007 Connie Rubirosa 2008 -- 2010 Michael Cutter Jack McCoy Appearances on other TV
shows ( edit ) Homicide : Life on the Street Season 6 Episode 5 : `` Baby ,
</item 3>

<item 4>
Jack McCoy
By the end of the 18th season , however , McCoy discovers that Shalvoy is involved in a prostitution scandal that is tied to a murder case he is prosecuting .
Angered and disappointed , McCoy orders Cutter to start investigating Shalvoy , who retaliates by lending his support to McCoy 's opponent in the election . In
the last episode of season 19 , `` The Drowned and the Saved '' , Cutter uncovers evidence that Shalvoy tried to buy a Senate seat for his wife , and threatens
him with public disgrace unless he resigns . McCoy 's opponent suddenly has no patron , giving McCoy good prospects for victory . In the opening episode of
season 20 , McCoy was revealed to have won the election ; he serves as DA for the remainder of this final season of Law & Order . Personal Life ( edit ) While
he is a brilliant legal mind , McCoy has more than a few personal demons . He was abused by his father , an Irish Chicago policeman who had also beat Jack 's
mother , and who eventually died of cancer . McCoy says that his determination and unyielding work ethic are a byproduct of having been harshly punished by his
father for losing at anything .
</item 4>

<item 5>
Jack McCoy
His underlying motivation , however , is not , he maintains , corruption , but a sincere desire to see justice done . To that end , McCoy has gone after
defendants accused of perverting the justice system to arrange wrongful convictions with just as much determination as his more mundane cases . Such aggressive
actions in the courts have earned him the nickname `` Hang ' em High McCoy '' . He has subsequently developed a reputation with both colleagues and rival
attorneys , once being referred to as `` the top of the legal food chain '' by a rival attorney during a trial . Following the 17th season ( 2006 -- 2007 ) ,
Jack McCoy was appointed interim district attorney , taking over from Arthur Branch ( Fred Thompson ) . McCoy 's appearance on Law & Order : Special Victims
Unit on the November 13 , 2007 , episode `` Blinded '' , marked his first appearance in the Law & Order universe as district attorney . The replacement for his
former position is Michael Cutter ( Linus Roache ) , a prosecutor with a penchant for recklessness not unlike McCoy 's own in his younger days . This
occasionally presents political difficulties for the new district attorney .
</item 5>

<item 6>
Jack McCoy
1 Season one 4.1. 2 Season two 4.2 As district attorney 4.2. 1 Season 9 4.2. 2 Season 10 4.2. 3 Season 11 4.2. 4 Season 12 4.2. 5 Season 13 4.2. 6 Season 19 5
Reception 6 District attorney 's office timeline 7 Appearances on other TV shows 8 References Character overview ( edit ) Jack McCoy brings 24 years of
experience with him as he is appointed executive assistant district attorney by Adam Schiff ( Steven Hill ) in the season - five premiere episode `` Second
Opinion '' . He quickly establishes himself as a more unconventional , ruthless litigator than his predecessor , Ben Stone ( Michael Moriarty ) . He often bends
-- and sometimes breaks -- trial rules to get convictions , finds tenuous rationales for charging defendants with crimes when the original charges fail to stick
, and charges innocent people to frighten them into testifying against others . McCoy is found in contempt of court 80 times for such behavior , and his tactics
occasionally incur negative publicity for the DA 's office .
</item 6>

<item 7>
Graph query row:
person: Jack McCoy
episode: Second Opinion
date: September 21, 1994
relationship: RELATED_TO
</item 7>

<item 8>
Jack McCoy
He was created by Michael S. Chernuchin and portrayed by Sam Waterston from 1994 until the end of the series in 2010 . He is the second - longest tenured
character on the show ( 16 seasons ) , after Lt. Anita Van Buren ( 17 seasons ; portrayed by S. Epatha Merkerson ) . He appeared in 368 episodes of Law & Order
, four episodes of Law & Order : Special Victims Unit , two episodes of Law & Order : Trial by Jury , two episodes of Homicide : Life on the Street , and the
made - for - TV movie Exiled . Waterston 's performance as McCoy on the New York - based series was so popular that it resulted in him being declared a ``
Living Landmark '' by the New York Landmarks Conservancy , along with fellow longtime series cast member Jerry Orbach ( who portrayed the popular police
detective Lennie Briscoe for 12 years ) . Contents ( hide ) 1 Character overview 2 Personal life 3 Notable conflicts 4 In Law & Order : Special Victims Unit 4.1
As executive assistant district attorney 4.1.
</item 8>

<item 9>
Jack McCoy
Although McCoy insists that he is a prosecutor , and not a politician , McCoy replaces Branch as the DA immediately thereafter . In Law & Order : Special
Victims Unit ( edit ) McCoy has appeared in four episodes of Law & Order : Special Victims Unit ; however , he is often referred to in the series , and his
actions affect the ADAs working with the Special Victims Unit . As executive assistant district attorney ( edit ) Season one ( edit ) `` Entitled '' ( onscreen
) : McCoy makes his first appearance on the series . McCoy , working with his assistant , Abbie Carmichael ( Angie Harmon ) , assists the Special Victims Unit
in solving a cold case , that Briscoe handled years ago with his former partner , Detective Mike Logan ( Chris Noth ) . Season two ( edit ) `` Baby Killer '' (
mention ) : After a murder intertwines with ADA Alexandra Cabot 's ( Stephanie March ) sex - crimes case , Chief Assistant District Attorney Charlie Phillips (
Jeffrey DeMunn ) , Lewin 's second in command , hands the murder case to McCoy . As district attorney ( edit ) Season 9 ( edit ) `` Blinded '' ( onscreen ) :
McCoy had been promoted to district attorney .
</item 9>

<item 10>
Jack McCoy
Jack McCoy - wikipedia Jack McCoy Jump to : navigation , search For the politician and labor activist , see Jack E. McCoy . Jack McCoy Law & Order character
First appearance `` Second Opinion '' `` Entitled '' ( SVU ) Last appearance `` Rubber Room '' ( Law & Order ) `` The Undiscovered Country '' ( SVU ) Portrayed
by Sam Waterston Time on show 1994 -- 2010 , 2018 Seasons 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 Credited appearances 368
episodes ( L&O ) 2 episodes ( HLOTS ) 4 episodes ( SVU ) 2 episodes ( TBJ ) 376 episodes ( total ) Preceded by Ben Stone ( EADA ) Arthur Branch ( DA ) ( L&O )
Succeeded by Michael Cutter ( EADA ) Information Family Rebecca McCoy ( daughter ) John James McCoy is a fictional character in the television drama Law & Order
.
</item 10>
=== END ITEM id=76a164a70041a2bc ===
