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

=== ITEM id=642916a19e05d401 ===
Gold sentences and facts:
Sentence 0: `` You 'll Never Find Another Love Like Mine '' ( written by Kenny Gamble & Leon Huff ) is a song performed by R&B singer Lou Rawls on his 1976 album All Things in Time . The song proved to be Rawls ' breakthrough hit , reaching number one on both the R&B and Easy Listening charts as well as number four on the dance chart and number two on the US Billboard Hot 100 , where it was kept from the top spot for two weeks by `` You Should Be Dancing '' by The Bee Gees and `` ( Shake , Shake , Shake ) Shake Your Booty '' by KC and the Sunshine Band , respectively . This was the first and only time that one of Rawls ' records managed to reach Billboard 's pop top ten . It was the first big hit for Philadelphia International to feature the reformulated MFSB , after many of the original members left Gamble and Huff for better opportunities . The song started Rawls ' live shows from 1977 on .
  Fact 0.0: "You'll Never Find Another Love Like Mine" was written by Kenny Gamble and Leon Huff.
  Fact 0.1: "You'll Never Find Another Love Like Mine" is a song performed by R&B singer Lou Rawls.
  Fact 0.2: "You'll Never Find Another Love Like Mine" is on Lou Rawls' 1976 album All Things in Time.
  Fact 0.3: "You'll Never Find Another Love Like Mine" proved to be Rawls' breakthrough hit.
  Fact 0.4: "You'll Never Find Another Love Like Mine" reached number one on the R&B chart.
  Fact 0.5: "You'll Never Find Another Love Like Mine" reached number one on the Easy Listening chart.
  Fact 0.6: "You'll Never Find Another Love Like Mine" reached number four on the dance chart.
  Fact 0.7: "You'll Never Find Another Love Like Mine" reached number two on the US Billboard Hot 100.
  Fact 0.8: On the Billboard Hot 100, "You'll Never Find Another Love Like Mine" was kept from the top spot for two weeks by "You Should Be Dancing" by The Bee Gees and by "(Shake, Shake, Shake) Shake Your Booty" by KC and the Sunshine Band, respectively.
  Fact 0.9: "You'll Never Find Another Love Like Mine" was the first and only time one of Rawls' records managed to reach Billboard's pop top ten.
  Fact 0.10: "You'll Never Find Another Love Like Mine" was the first big hit for Philadelphia International to feature the reformulated MFSB.
  Fact 0.11: Many of the original MFSB members left Gamble and Huff for better opportunities.
  Fact 0.12: The song started Lou Rawls' live shows from 1977 on.

Retrieved items:
<item 1>
You'll Never Find Another Love Like Mine
Cover versions ( edit ) `` You 'll Never Find Another Love Like Mine '' has been covered by singers Michael Bublé , Laura Pausini , reggae singer John Holt and
by the Dub Pistols ( who use a sample of John Holt 's version ) on their Speakers and Tweeters album . Jazz saxophonist Stanley Turrentine covered the song on
his 1976 album The Man with the Sad Face . as did Hank Crawford on Hank Crawford 's Back . This song was also featured on The Proud Family Soundtrack . See also
( edit ) List of number - one R&B singles of 1976 ( U.S. ) List of number - one adult contemporary singles of 1976 ( U.S. ) References ( edit ) Jump up ^
Whitburn , Joel ( 2004 ) . Top R&B / Hip - Hop Singles : 1942 - 2004 . Record Research . p. 483 . Jump up ^ Whitburn , Joel ( 2004 ) . Hot Dance / Disco : 1974
- 2003 . Record Research . p. 213 .
</item 1>

<item 2>
Lou Rawls’s “You’ll Never Find Another Love Like Mine,” written and produced by Kenny Gamble and Leon Huff, was released in 1976 on *All Things in Time*. It
topped the R&B and Easy Listening charts, reached No. 2 on the Billboard Hot 100, sold over a million copies, and became his only pop Top 10 hit. It was widely
covered and featured in films and television. Spanky and Our Gang’s 1967 sunshine-pop single “
</item 2>

<item 3>
You'll Never Find Another Love Like Mine
The song proved to be Rawls ' breakthrough hit , reaching number one on both the R&B and Easy Listening charts as well as number four on the dance chart and
number two on the US Billboard Hot 100 , where it was kept from the top spot for two weeks by `` You Should Be Dancing '' by The Bee Gees and `` ( Shake , Shake
, Shake ) Shake Your Booty '' by KC and the Sunshine Band , respectively . This was the first and only time that one of Rawls ' records managed to reach
Billboard 's pop top ten . It was the first big hit for Philadelphia International to feature the reformulated MFSB , after many of the original members left
Gamble and Huff for better opportunities . The song started Rawls ' live shows from 1977 on . The single went on to sell over a million copies and was certified
gold by the RIAA . Contents ( hide ) 1 Chart performance 1.1 Weekly charts 1.
</item 3>

<item 4>
You'll Never Find Another Love Like Mine
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 4>

<item 5>
You'll Never Find Another Love Like Mine
You 've Heard It All Let Me Be Good to You Sit Down and Talk to Me Singles `` Love Is a Hurtin ' Thing '' `` Your Good Thing ( Is About to End ) '' `` Bring It
On Home to Me '' `` You 'll Never Find Another Love Like Mine '' `` Lady Love '' Retrieved from ``
https://en.wikipedia.org/w/index.php?title=You%27ll_Never_Find_Another_Love_Like_Mine&oldid=803225603 '' Categories : 1976 singles 1976 songs Billboard Adult
Contemporary number - one singles Billboard Hot R&B / Hip - Hop Songs number - one singles Lou Rawls songs Philadelphia International Records singles Songs
written by Kenny Gamble Songs written by Leon Huff Songs from animated films Hidden categories : All articles with dead external links Articles with dead
external links from July 2016 Articles with permanently dead external links Articles with hAudio microformats Talk Contents About Wikipedia Italiano Português
Edit links This page was last edited on 1 October 2017 , at 07 : 04 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional
terms may apply .
</item 5>

<item 6>
You'll Never Find Another Love Like Mine
Record Research . p. 213 . Jump up ^ `` Billboard Hot 100 '' . 1976 - 09 - 04 . Retrieved 2015 - 06 - 07 . Jump up ^ `` Billboard Hot 100 '' . 1976 - 09 - 11 .
Retrieved 2015 - 06 - 07 . Jump up ^ `` The Hits Just Keep On Comin ' : You 'll Never Find Another Love Like Mine '' . Hitsjustkeeponcomin.blogspot.com. 2005 -
01 - 12 . Retrieved 2016 - 10 - 12 . Jump up ^ ( 1 ) Jump up ^ `` Archived copy '' . Archived from the original on June 2 , 2016 . Retrieved October 27 , 2015 .
Jump up ^ `` Image : RPM Weekly - Library and Archives Canada '' . Bac-lac.gc.ca . Retrieved 2016 - 10 - 12 . Jump up ^ Whitburn , Joel ( 2002 ) . Top Adult
Contemporary : 1961 - 2001 . Record Research .
</item 6>

<item 7>
You'll Never Find Another Love Like Mine
You 'll Never Find Another Love Like Mine - wikipedia You 'll Never Find Another Love Like Mine Jump to : navigation , search `` You 'll Never Find Another Love
Like Mine '' Single by Lou Rawls from the album All Things in Time B - side `` Let 's Fall in Love All Over Again '' Released May 1976 Format 7 '' Genre R&B ,
jazz , disco Length 4 : 26 ( album version ) 3 : 36 ( single edit ) Label Philadelphia International Records Songwriter ( s ) Kenny Gamble Leon Huff Producer (
s ) Kenny Gamble Leon Huff Lou Rawls singles chronology `` A Natural Man '' ( 1971 ) `` You 'll Never Find Another Love Like Mine '' ( 1976 ) `` Groovy People
'' ( 1976 ) `` A Natural Man '' ( 1971 ) `` You 'll Never Find Another Love Like Mine '' ( 1976 ) `` Groovy People '' ( 1976 ) `` You 'll Never Find Another
Love Like Mine '' ( written by Kenny Gamble & Leon Huff ) is a song performed by R&B singer Lou Rawls on his 1976 album All Things in Time .
</item 7>

<item 8>
You'll Never Find Another Love Like Mine
Top Adult Contemporary : 1961 - 2001 . Record Research . p. 200 . Jump up ^ `` Top Singles -- Volume 26 , No. 14 & 15 , January 08 1977 '' . RPM . Library and
Archives Canada . Retrieved June 13 , 2016 . Jump up ^ `` Top 100 Hits of 1976 / Top 100 Songs of 1976 '' . Musicoutfitters.com . Retrieved 2016 - 10 - 12 .
Jump up ^ Erlewine , Michael . The Man with the Sad Face -- Review at AllMusic . Retrieved April 16 , 2015 . External links ( edit ) Lyrics of this song at
MetroLyrics ( hide ) Lou Rawls Albums Stormy Monday All Things in Time Unmistakably Lou When You Hear Lou ,
</item 8>

<item 9>
You'll Never Find Another Love Like Mine
Contents ( hide ) 1 Chart performance 1.1 Weekly charts 1.2 Year - end charts 2 Later uses 3 Cover versions 4 See also 5 References 6 External links Chart
performance ( edit ) Weekly charts ( edit ) Chart ( 1976 -- 77 ) Peak position Australia KMR 22 Canada RPM Top Singles UK 10 US Billboard Hot 100 US Billboard
Hot Dance Club Play US Billboard Hot Adult Contemporary Tracks US Billboard Hot Black Singles Year - end charts ( edit ) Chart ( 1976 ) Rank Canada 38 US
Billboard Hot 100 32 Chart ( 1977 ) Rank Australia 97 Later uses ( edit ) It can also be heard in the films The Hot Chick ( 2002 ) , Guess Who ( 2005 ) ,
Disturbia ( 2007 ) , Ice Age : Dawn of the Dinosaurs ( 2009 ) , Hit and Run ( 2012 ) , Veronica Mars ( 2014 ) , and on the TV sitcoms My Wife and Kids ( 2001 --
2005 ) , in which Lou Rawls himself sings it to Damon Wayans during a colonoscopy ,
</item 9>

<item 10>
You'll Never Find Another Love Like Mine
in which Lou Rawls himself sings it to Damon Wayans during a colonoscopy , That ' 70s Show ( 1998 -- 2006 ) , South Park ( 1997 -- ) , Two and a Half Men ( 2003
-- 2015 ) , and Psych ( 2006 -- 2014 ) . The song could also be heard in an episode of The Proud Family , where Lou Rawls sings and dances with Penny Proud . In
addition , it was performed by the singer in doll form in an Action League Now ! segment on KaBlam ! . The song `` If You Could Love Me '' by Edwyn Collins uses
many of the same chords , and has a similar sound . Rocco DiSpirito and Karina Smirnoff also danced the rumba to this song in week 2 of season 7 of Dancing with
the Stars .
</item 10>
=== END ITEM id=642916a19e05d401 ===

=== ITEM id=7aa35530811afca9 ===
Gold sentences and facts:
Sentence 0: The film was directed by Robert Mulligan , and starred Gary Grimes as Hermie , Jerry Houser as his best friend Oscy , Oliver Conant as their nerdy young friend Benjie , Jennifer O'Neill as Hermie 's mysterious love interest , and Katherine Allentuck and Christopher Norris as a pair of girls whom Hermie and Oscy attempt to seduce . Mulligan also has an uncredited role as the voice of the adult Hermie . Maureen Stapleton ( Allentuck 's real - life mother ) also appears in a small , uncredited voice role ( calling after Hermie as he leaves the house in an early scene , and after he enters his room in a later scene ) .
  Fact 0.0: Robert Mulligan directed the film.
  Fact 0.1: Gary Grimes starred as Hermie.
  Fact 0.2: Jerry Houser starred as Hermie's best friend Oscy.
  Fact 0.3: Oliver Conant played Benjie, a nerdy young friend of Hermie and Oscy.
  Fact 0.4: Jennifer O'Neill played Hermie's mysterious love interest.
  Fact 0.5: Katherine Allentuck and Christopher Norris played a pair of girls whom Hermie and Oscy attempt to seduce.
  Fact 0.6: Robert Mulligan has an uncredited role as the voice of the adult Hermie.
  Fact 0.7: Maureen Stapleton is Katherine Allentuck's real-life mother.
  Fact 0.8: Maureen Stapleton has a small, uncredited voice role.
  Fact 0.9: Maureen Stapleton's voice calls after Hermie as he leaves the house in an early scene.
  Fact 0.10: Maureen Stapleton's voice calls after Hermie as he enters his room in a later scene.
Sentence 1: Summer of ' 42 Theatrical release poster Directed by Robert Mulligan Produced by Richard A. Roth Written by Herman Raucher Starring Jennifer O'Neill Gary Grimes Jerry Houser Oliver Conant Narrated by Robert Mulligan Music by Michel Legrand Cinematography Robert Surtees Edited by Folmar Blangsted Distributed by Warner Bros. Pictures Release date April 18 , 1971 ( 1971 - 04 - 18 ) Running time 104 minutes Country United States Language English Budget $1 million Box office $32,063,634
  Fact 1.0: Summer of '42 has a theatrical release poster.
  Fact 1.1: Summer of '42 was directed by Robert Mulligan.
  Fact 1.2: Summer of '42 was produced by Richard A. Roth.
  Fact 1.3: Summer of '42 was written by Herman Raucher.
  Fact 1.4: Summer of '42 starred Jennifer O'Neill, Gary Grimes, Jerry Houser and Oliver Conant.
  Fact 1.5: Summer of '42 was narrated by Robert Mulligan.
  Fact 1.6: Summer of '42 had music by Michel Legrand.
  Fact 1.7: Summer of '42 had cinematography by Robert Surtees.
  Fact 1.8: Summer of '42 was edited by Folmar Blangsted.
  Fact 1.9: Summer of '42 was distributed by Warner Bros. Pictures.
  Fact 1.10: Summer of '42 was released on April 18, 1971.
  Fact 1.11: Summer of '42 has a running time of 104 minutes.
  Fact 1.12: Summer of '42 is a United States production.
  Fact 1.13: Summer of '42 is in English.
  Fact 1.14: Summer of '42 had a budget of $1 million.
  Fact 1.15: Summer of '42 had a box office of $32,063,634.

Retrieved items:
<item 1>
Graph query row:
person: Gary Grimes
film: Summer of '42
relationship: RELATED_TO
role: Hermie
</item 1>

<item 2>
Summer of '42
Summer of ' 42 - wikipedia Summer of ' 42 Jump to : navigation , search Summer of ' 42 Theatrical release poster Directed by Robert Mulligan Produced by Richard
A. Roth Written by Herman Raucher Starring Jennifer O'Neill Gary Grimes Jerry Houser Oliver Conant Narrated by Robert Mulligan Music by Michel Legrand
Cinematography Robert Surtees Edited by Folmar Blangsted Distributed by Warner Bros. Pictures Release date April 18 , 1971 ( 1971 - 04 - 18 ) Running time 104
minutes Country United States Language English Budget $1 million Box office $32,063,634 Summer of ' 42 is a 1971 American coming - of - age comedy - drama film
based on the memoirs of screenwriter Herman Raucher . It tells the story of how Raucher , in his early teens on his 1942 summer vacation on Nantucket Island (
off the coast of Cape Cod ) , embarks on a one - sided romance with a young woman , Dorothy , whose husband had gone off to fight in World War II .
</item 2>

<item 3>
Summer of '42
The film is recognized by American Film Institute in these lists : 2002 : AFI 's 100 Years ... 100 Passions -- Nominated 2005 : AFI 's 100 Years of Film Scores
-- Nominated Sequel ( edit ) In 1973 , the film was followed by Class of ' 44 , a slice - of - life film made up of vignettes about Herman Raucher and Oscar
Seltzer 's experiences in college . Class of ' 44 involves the boys facing army service in the last year of World War II . The only crew member from Summer of '
42 to return to the project was Raucher himself , who wrote the script ; a new director and composer were brought in to replace Mulligan and Legrand . Of the
principal four cast members of Summer of ' 42 , only Jerry Houser and Gary Grimes returned for prominent roles , with Oliver Conant making two brief appearances
totaling less than two minutes of screen time . Jennifer O'Neill did not appear in the film at all , nor was the character of Dorothy mentioned . The film is
noted for featuring a young , slim John Candy briefly appearing in his first film role .
</item 3>

<item 4>
Summer of '42
Retrieved September 3 , 2010 . Jump up ^ Internet Movie Database : Summer of ' 42 Business Data IMDb.com Retrieved July 3 , 2006 . Jump up ^ `` Summer of ' 42 (
1971 ) -- Misc Notes '' . TCM.com . Retrieved February 8 , 2012 . Jump up ^ `` AFI 's 100 Years ... 100 Passions Nominees '' ( PDF ) . Retrieved August 20 ,
2016 . Jump up ^ `` AFI 's 100 Years of Film Scores Nominees '' ( PDF ) . Retrieved August 20 , 2016 .
</item 4>

<item 5>
Summer of '42
Archived from the original on January 29 , 2009 . Jump up ^ Jennifer O'Neill in 2002 tv-now.com Retrieved August 11 , 2006 Archived August 29 , 2006 , at the
Wayback Machine . Jump up ^ Theatermania.com , Starry , Starry Morning : Charles Nelson 's Casts and Forecasts . Retrieved September 3 , 2007 Jump up ^ Summer
of ' 42 Plus Twenty usatoday.com . Retrieved July 6 , 2006 . External links ( edit ) www.hermanraucher.com Summer of ' 42 on IMDb Summer of ' 42 at the TCM
Movie Database Summer of ' 42 by Andy Williamson -- The Wordslinger dated April 14 ,
</item 5>

<item 6>
Graph query row:
person: Jennifer O'Neill
film: Summer of '42
relationship: RELATED_TO
role: Dorothy
</item 6>

<item 7>
Summer of '42
wikipedia.org/w/index.php?title=Summer_of_%2742&oldid=803913855 '' Categories : 1971 films English - language films 1970s comedy - drama films 1970s coming - of
- age films 1970s teen films American biographical films American comedy - drama films American coming - of - age films American films Film scores by Michel
Legrand Films about vacationing Films about virginity Films directed by Robert Mulligan Films set in 1942 Films set in Massachusetts Films set on beaches Films
set on islands Films set on the home front during World War II Films that won the Best Original Score Academy Award Nantucket , Massachusetts Warner Bros.
</item 7>

<item 8>
Summer of '42
Cast ( edit ) Jennifer O'Neill as Dorothy Gary Grimes as Hermie Jerry Houser as Oscy Oliver Conant as Benjie Katherine Allentuck as Aggie Christopher Norris as
Miriam Lou Frizzell as Druggist Robert Mulligan ( uncredited voice ) as Narrator / Older Herman Raucher Walter Scott ( uncredited ) as Dorothy 's husband
Maureen Stapleton ( uncredited voice ) as Hermie 's Mother Production ( edit ) Herman Raucher wrote the film script in the 1950s during his tenure as a
television writer , but `` could n't give it away . '' In the 1960s , he met Robert Mulligan , best known for directing To Kill a Mockingbird . Raucher showed
Mulligan the script , and Mulligan took it to Warner Bros. , where Mulligan argued the film could be shot for the relatively low price of $1 million , and
Warner approved it . They had so little faith in the film becoming a box - office success , though , they shied from paying Raucher outright for the script ,
instead promising him ten percent of the gross . When casting for the role of Dorothy , Warner Bros.
</item 8>

<item 9>
Graph query row:
person: Jerry Houser
film: Summer of '42
relationship: RELATED_TO
role: Oscy
</item 9>

<item 10>
Graph query row:
person: Oliver Conant
film: Summer of '42
relationship: RELATED_TO
role: Benjie
</item 10>
=== END ITEM id=7aa35530811afca9 ===

=== ITEM id=21b03421367d3adc ===
Gold sentences and facts:
Sentence 0: On May 1 , 2017 , Slasher began filming a second season in Orangeville , Ontario , Canada . Shaftesbury Films confirmed that the series will not be returning to Super Channel or Chiller for a second season . However , Netflix acquired the rights to the second season . The second season was released via Netflix on October 17 , 2017 .
  Fact 0.0: Slasher began filming a second season on May 1, 2017.
  Fact 0.1: Slasher filmed its second season in Orangeville, Ontario, Canada.
  Fact 0.2: Shaftesbury Films confirmed that Slasher will not return to Super Channel or Chiller for a second season.
  Fact 0.3: Netflix acquired the rights to the second season of Slasher.
  Fact 0.4: The second season of Slasher was released via Netflix on October 17, 2017.
Sentence 1: The first season , retroactively subtitled The Executioner , was produced in association with the Canadian network Super Channel , Slasher was the first original series by U.S. TV channel Chiller , which premiered the series on Friday , March 4 , 2016 , at 9 : 00 pm EST . Super Channel aired the show 's Canadian premiere on April 1 , 2016 . The series first season , which centered on a mysterious figure billed as `` The Executioner '' terrorizing the fictional town of Waterbury , Canada , was filmed between July and October 2015 in Sudbury , Parry Sound , and Sault Ste . Marie , Ontario .
  Fact 1.0: The first season of Slasher, retroactively subtitled The Executioner, was produced in association with the Canadian network Super Channel.
  Fact 1.1: Slasher was the first original series by U.S. TV channel Chiller.
  Fact 1.2: Chiller premiered the series on Friday, March 4, 2016, at 9:00 pm EST.
  Fact 1.3: Super Channel aired the show's Canadian premiere on April 1, 2016.
  Fact 1.4: The first season of Slasher centered on a mysterious figure billed as 'The Executioner'.
  Fact 1.5: The Executioner terrorized the fictional town of Waterbury, Canada.
  Fact 1.6: The first season was filmed between July and October 2015.
  Fact 1.7: The first season was filmed in Sudbury, Parry Sound, and Sault Ste. Marie, Ontario.

Retrieved items:
<item 1>
Slasher (TV series)
A first trailer for the series would be released on November 26 , 2015 . Unlike most TV series that film their episodes in order , Slasher , under the direction
of Craig David Wallace , was shot as if it were a `` super-sized '' movie : Scenes from multiple episodes were shot at the same time , with the availability of
locations and cast being factored in . The out - of - order schedule allowed the actors to know of their characters ' fates , especially those who had to film
their death scenes one day but return later to film earlier scenes as necessary . As an example of this , Martin cited Mark Ghanimé 's first day on set , when
his character , Justin Faysal , was laid out in a casket for a scene early in Slasher 's third episode ( Justin 's death , which took place in Episode 2 , would
be filmed later on ) . On May 1 , 2017 , Slasher began filming a second season in Orangeville , Ontario , Canada . Shaftesbury Films confirmed that the series
will not be returning to Super Channel or Chiller for a second season .
</item 1>

<item 2>
Slasher (TV series)
Contents ( hide ) 1 Background 2 Plot 2.1 Season 1 : The Executioner ( 2016 ) 2.2 Season 2 : Guilty Party ( 2017 ) 3 Cast and characters 3.1 Recurring cast
members 3.2 Season 1 : The Executioner 3.2. 1 Main 3.2. 2 Recurring 3.3 Season 2 : Guilty Party 4 Episodes 4.1 Season 1 : The Executioner ( 2016 ) 4.2 Season 2
: Guilty Party ( 2017 ) 5 Development and production 6 Reception 6.1 Awards and accolades 7 International broadcast 8 References 9 External links Background (
edit ) Slasher employs a season - long anthology format , with an overarching mystery storyline that will be resolved upon the season 's conclusion .
</item 2>

<item 3>
Slasher (TV series)
The Futon Critic . July 28 , 2016 . Retrieved March 6 , 2016 . ^ Jump up to : `` Lights , camera , action ... ' Slasher ' TV series begins filming in
Orangeville '' . Orangeville Banner . May 1 , 2017 . Retrieved May 2 , 2017 . ^ Jump up to : `` Slasher returns for a second season '' . Playback Online . May 9
, 2017 . Retrieved September 29 , 2017 . Jump up ^ Moore , Kasey ( September 20 , 2017 ) . `` October 2017 New Netflix Releases '' . WhatsonNetlix . Retrieved
September 20 , 2017 . Jump up ^ Urqyhart - White , Alaina ( October 17 , 2017 ) . `` Is ' Slasher 2 : Guilty Party ' A True Story ? The Netflix Series Honors
Some Legendary Horror Classics '' . Bustle . Retrieved October 18 , 2017 . ^ Jump up to : Patrick Hipes .
</item 3>

<item 4>
Slasher (TV series)
of seasons No. of episodes 16 ( list of episodes ) Production Executive producer ( s ) Aaron Martin Greg Phillips Jonathan Ford Christina Jennings Scott Garvie
Location ( s ) Ontario , Canada Editor ( s ) Erin Deck Camera setup Single - camera Running time 46 - 53 minutes Production company ( s ) Shaftesbury Films
Super Channel ( season 1 ) TVA Group ( season 2 ) Distributor Content Media Corporation Release Original network Super Channel ( Canada , season 1 ) Chiller (
USA , season 1 ) Netflix ( season 2 ) Original release March 4 , 2016 ( 2016 - 03 - 04 ) -- present External links Website www.netflix.com/title/80105452
Slasher is a Canadian television horror anthology series created by Aaron Martin . The first season , retroactively subtitled The Executioner , was produced in
association with the Canadian network Super Channel , Slasher was the first original series by U.S. TV channel Chiller , which premiered the series on Friday ,
March 4 , 2016 , at 9 : 00 pm EST . Super Channel aired the show 's Canadian premiere on April 1 , 2016 .
</item 4>

<item 5>
Slasher (TV series)
Slasher ( TV series ) - wikipedia Slasher ( TV series ) Jump to : navigation , search Slasher Genre Anthology Horror Slasher Drama Created by Aaron Martin
Starring Katie McGrath Brandon Jay McLaren Steve Byers Patrick Garrow Dean McDermott Leslie Hope Lovell Adams - Gray Jim Watson Christopher Jacot Paulino Nunes
Ty Olsson Joanne Vannicola Sebastian Pigott Madison Cheeatow Melinda Shankar Kaitlyn Leeb Rebecca Liddiard Paula Brancati Composer ( s ) Shawn Pierce Country of
origin Canada Original language ( s ) English No. of seasons No.
</item 5>

<item 6>
Slasher (TV series)
Not wanting to have the show 's killer be `` a mythological creature '' ( as he feels the killers in most slasher films do not have much mystery surrounding
them ) , Martin also uses elements of the traditional whodunit in Slasher : The show 's characters , many of whom have mysterious backgrounds -- and their own
reasons for possibly being the killer -- are featured , explored , and eliminated from consideration , one by one ( either through death or the natural
deductive process ) , until the `` all too human '' killer and their motivations for their actions are revealed . Plot ( edit ) Season 1 : the Executioner (
2016 ) ( edit ) Slasher 's 8 - episode first season centers around Sarah Bennett who , with husband Dylan , moves back to the town of her birth , Waterbury ( a
fictional town in Canada ) , and into her parents ' former home . It was in that house on Halloween Night 1988 where both of her parents , Bryan and Rachel ,
were murdered . Rachel was pregnant with Sarah at the time of her murder , with police discovering the killer holding Rachel 's newborn baby after the slayings
.
</item 6>

<item 7>
Graph query row:
series: Filming of Slasher's first season
relationship: LOCATED_IN
location: Northern Ontario
</item 7>

<item 8>
Slasher (TV series)
Awards and accolades ( edit ) Year Award Nominee ( s ) Result 2017 5th Annual Canadian Screen Awards Best TV Movie or Limited Series Slasher Nominated Best
Direction in a Dramatic Program or Limited Series Nominated Best Writing in a Dramatic Program or Limited Series Nominated Best Performance by an Actor in a
Leading Role in a Dramatic Program or Limited Series Steve Byers Nominated Best Performance by an Actress in a Featured Supporting Role in a Dramatic Program or
Series Wendy Crewson Won International broadcast ( edit ) On May 25 , 2016 , the entire first season of Slasher became available to stream instantly on Netflix
US . On October 17 , 2017 , the full second season of Slasher became available to stream on Netflix in various countries . References ( edit ) Jump up ^
Brzoznowski , Kristin . `` Super Channel Sets April Premiere for Original Series Slasher '' . TVDrama.ws . World Screen ( WSN Inc . ) . Retrieved 6 March 2016 .
^ Jump up to : `` Chiller and Shaftesbury Announce Start of Production and Cast for `` Slasher '' `` .
</item 8>

<item 9>
Slasher (TV series)
Season 2 : Guilty Party ( 2017 ) ( edit ) Slasher 's 8 - episode second season centers around a horrific secret that a group of former summer camp counselors
have buried long ago . Due to a changing circumstance , they must return to their resort in the dead of winter to retrieve the body of Talvinder , whom they
murdered while they were camp counsellors , five years earlier . They get surprised when they ca n't find the body at the campsite . Later , the group start
dying one by one at the hands of a mysterious person wearing a parka , which also alerts the second group of counselors at the resort . Eventually , it 's
revealed that Judith is the killer , and another character , Wren , is actually Owen `` Wren '' Turnbull , her son who committed suicide after being framed for
Talvinder 's murder , and is a figment of Judith 's imagination . Owen wrote a letter to his mother , asking her to get justice for him before hanging himself .
</item 9>

<item 10>
Slasher (TV series)
Super Channel aired the show 's Canadian premiere on April 1 , 2016 . The series first season , which centered on a mysterious figure billed as `` The
Executioner '' terrorizing the fictional town of Waterbury , Canada , was filmed between July and October 2015 in Sudbury , Parry Sound , and Sault Ste . Marie
, Ontario . On May 1 , 2017 , Slasher began filming a second season in Orangeville , Ontario , Canada . The licensing rights for the second season was acquired
by Netflix . The season was released as a `` Netflix Original '' title in various countries on October 17 , 2017 . The second season titled Slasher : Guilty
Party is centered on a group of former summer camp counselors who return to an isolated campground in order to bury their sins of the past , before being
targeted , one by one , by an unknown killer .
</item 10>
=== END ITEM id=21b03421367d3adc ===

=== ITEM id=5b0ed58ebe4b4612 ===
Gold sentences and facts:
Sentence 0: Year Film Role Notes Dumb and Dumber State Trooper Down Periscope E.T. `` Sonar '' Lovacelli RocketMan Fred Z . Randall Wag the Dog Pet Wrangler Space Ghost Coast to Coast Himself 1 episode 1998 Half Baked Kenny Davis There 's Something About Mary Hitchhiker ( uncredited ) Dog Park Callum Ned 's Newt Newton ( voice ) Mr. Headmistress Tucker 1999 Superstar Slater 2000 The Whole Nine Yards Special Agent Steve Hanson Big Money Hustlas Officer Harry Cox Becoming Dick Richard Breggs Freddy Got Fingered Darren Gary & Mike Mike Bonner ( voice ) 2002 Back by Midnight Sheriff Hubbard Sorority Boys Doofer / Roberta 2003 Family Tree Jake Kart Racer Zee 2005 Lucky 13 Bleckman Because of Winn - Dixie Policeman Robots Lugnut ( voice ) 2006 Employee of the Month Russell Porpis - Gunders Surf School Rip Ergo Proxy Head Dog Cop ( voice ) 1 episode 2007 Meet the Robinsons Carl ( voice ) The Dukes of Hazzard : The Beginning Sheriff Rosco P. Coltrane 2008 Bachelor Party 2 : The Last Temptation Derek Madagascar : Escape 2 Africa Giraffe ( voice ) 2009 Fudgy Wudgy Fudge Face Elmore P. Fudge My Life in Ruins Big Al Sawchuck 2009 The Haunted World of El Superbeasto Gerard the Exterminator ( voice ) 2011 Spooky Buddies Warwick the Warlock Lloyd the Conqueror Vulcan Dahmer Vs. Gacy God Dan Vs . Hiram Episode : Technology Robot Chicken Professor Cornelius Q. Quibblefingers ( voice ) 1 episode ; also writes 4 episodes 2012 The High Fructose Adventures of Annoying Orange Apple ( voice ) Robot and Monster Monster ( voice ) Main Role 2014 Back in the Day Skunk 2015 Jake and the Never Land Pirates Captain Frost ( voice ) Penn Zero : Part - Time Hero Helper Hue ( voice ) 2016 Sausage Party Ketchup Bottle / Drug Dealer / Baba Ganoush ( voice ) Skylanders Academy Hugo ( voice ) 2017 - present Puppy Dog Pals Bob ( voice ) Also show creator
  Fact 0.0: The listed performer had the role State Trooper in Dumb and Dumber.
  Fact 0.1: The listed performer appeared in Down Periscope, with no role given.
  Fact 0.2: The listed performer had the role 'Sonar' Lovacelli in E.T.
  Fact 0.3: The listed performer had the role Fred Z. Randall in RocketMan.
  Fact 0.4: The listed performer had the role Pet Wrangler in Wag the Dog.
  Fact 0.5: The listed performer had the role Himself in Space Ghost Coast to Coast, for 1 episode.
  Fact 0.6: In 1998, the listed performer had the role Kenny Davis in Half Baked.
  Fact 0.7: In 1998, the listed performer had the role Hitchhiker, uncredited, in There's Something About Mary.
  Fact 0.8: In 1998, the listed performer had the role Callum in Dog Park.
  Fact 0.9: In 1998, the listed performer had the role Newton, a voice role, in Ned's Newt.
  Fact 0.10: In 1998, the listed performer had the role Tucker in Mr. Headmistress.
  Fact 0.11: In 1999, the listed performer had the role Slater in Superstar.
  Fact 0.12: In 2000, the listed performer had the role Special Agent Steve Hanson in The Whole Nine Yards.
  Fact 0.13: In 2000, the listed performer had the role Officer Harry Cox in Big Money Hustlas.
  Fact 0.14: In 2000, the listed performer had the role Richard Breggs in Becoming Dick.
  Fact 0.15: In 2000, the listed performer had the role Darren in Freddy Got Fingered.
  Fact 0.16: In 2000, the listed performer had the role Mike Bonner, a voice role, in Gary & Mike.
  Fact 0.17: In 2002, the listed performer had the role Sheriff Hubbard in Back by Midnight.
  Fact 0.18: In 2002, the listed performer had the role Doofer / Roberta in Sorority Boys.
  Fact 0.19: In 2003, the listed performer had the role Jake in Family Tree.
  Fact 0.20: In 2003, the listed performer had the role Zee in Kart Racer.
  Fact 0.21: In 2005, the listed performer had the role Bleckman in Lucky 13.
  Fact 0.22: In 2005, the listed performer had the role Policeman in Because of Winn-Dixie.
  Fact 0.23: In 2005, the listed performer had the role Lugnut, a voice role, in Robots.
  Fact 0.24: In 2006, the listed performer had the role Russell Porpis-Gunders in Employee of the Month.
  Fact 0.25: In 2006, the listed performer had the role Rip in Surf School.
  Fact 0.26: In 2006, the listed performer had the role Head Dog Cop, a voice role, for 1 episode, in Ergo Proxy.
  Fact 0.27: In 2007, the listed performer had the role Carl, a voice role, in Meet the Robinsons.
  Fact 0.28: In 2007, the listed performer had the role Sheriff Rosco P. Coltrane in The Dukes of Hazzard: The Beginning.
  Fact 0.29: In 2008, the listed performer had the role Derek in Bachelor Party 2: The Last Temptation.
  Fact 0.30: In 2008, the listed performer had the role Giraffe, a voice role, in Madagascar: Escape 2 Africa.
  Fact 0.31: In 2009, the listed performer had the role Fudge Face Elmore P. Fudge in Fudgy Wudgy.
  Fact 0.32: In 2009, the listed performer had the role Big Al Sawchuck in My Life in Ruins.
  Fact 0.33: In 2009, the listed performer had the role Gerard the Exterminator, a voice role, in The Haunted World of El Superbeasto.
  Fact 0.34: In 2011, the listed performer had the role Warwick the Warlock, Lloyd the Conqueror, and Vulcan in Spooky Buddies.
  Fact 0.35: In 2011, the listed performer had the role God in Dahmer Vs. Gacy.
  Fact 0.36: In 2011, the listed performer had the role Robot in Dan Vs. Hiram, Episode: Technology.
  Fact 0.37: In 2011, the listed performer had the role Professor Cornelius Q. Quibblefingers, a voice role, for 1 episode, in Robot Chicken.
  Fact 0.38: The listed performer also wrote 4 episodes of Robot Chicken.
  Fact 0.39: In 2012, the listed performer had the role Apple, a voice role, in The High Fructose Adventures of Annoying Orange.
  Fact 0.40: In 2012, the listed performer had the role Monster, a voice role and a main role, in Robot and Monster.
  Fact 0.41: In 2014, the listed performer had the role Skunk in Back in the Day.
  Fact 0.42: In 2015, the listed performer had the role Captain Frost, a voice role, in Jake and the Never Land Pirates.
  Fact 0.43: In 2015, the listed performer had the role Helper Hue, a voice role, in Penn Zero: Part-Time Hero.
  Fact 0.44: In 2016, the listed performer had the role Ketchup Bottle / Drug Dealer / Baba Ganoush, a voice role, in Sausage Party.
  Fact 0.45: In 2016, the listed performer had the role Hugo, a voice role, in Skylanders Academy.
  Fact 0.46: From 2017 to the present, the listed performer had the role Bob, a voice role, in Puppy Dog Pals.
  Fact 0.47: The listed performer is also the creator of Puppy Dog Pals.

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
Locations ( edit ) The majority of the film was shot in and around Clinton , Louisiana . The street scenes are set in Atlanta , but filmed in the New Orleans
Central Business District , and the university scenes were shot on the campus of Louisiana State University . Reception ( edit ) Box office ( edit ) The film
was number one at the box office its opening weekend and grossed $30.7 million on 3,785 screens . It also had an adjusted - dollar rank of number 24 all - time
for August releases . The film eventually collected $111 million worldwide , although it was much less successful financially outside the United States .
Critical reception ( edit ) Roger Ebert gave the film one star , calling it a `` lame - brained , outdated wheeze '' and wondered that Burt Reynolds ' part in
the film is `` karma-wise ... the second half of what Smokey and the Bandit was the first half of '' . Ebert also named it the second - worst film of the year
and Richard Roeper named it the worst film of 2005 .
</item 2>

<item 3>
Harland Williams
Williams created the Disney Junior series Puppy Dog Pals that debuted in 2017 . Filmography ( edit ) Year Film Role Notes Dumb and Dumber State Trooper Down
Periscope E.T. `` Sonar '' Lovacelli RocketMan Fred Z . Randall Wag the Dog Pet Wrangler Space Ghost Coast to Coast Himself 1 episode 1998 Half Baked Kenny
Davis There 's Something About Mary Hitchhiker ( uncredited ) Dog Park Callum Ned 's Newt Newton ( voice ) Mr.
</item 3>

<item 4>
California Highway Patrol
`` I was upset there were n't four or five more rounds in there . After that , I ran and jumped in a ditch . The dumbest thing is , I still had the service
revolver in my hand . I was afraid when more police came they 'd think I was one of the gunman . So I put it behind me and said , ' They went that way . ' ''
Suspects Jack Twinning and Bobby Davis escaped , later abandoned their vehicle and then split up . For nine hours , officers blanketed the area searching for
the killers . Twinning broke into a house and briefly held a man hostage . Officers used tear gas before storming the house , but Twinning committed suicide
with the shotgun he stole from Frago . Davis was captured , stood trial and convicted on four counts of murder . He was sentenced to death , but in 1972 , the
California Supreme Court declared the death penalty to be cruel and unusual punishment and in 1973 , the court commuted his sentence to life in prison .
</item 4>

<item 5>
Harland Williams
Headmistress Tucker 1999 Superstar Slater 2000 The Whole Nine Yards Special Agent Steve Hanson Big Money Hustlas Officer Harry Cox Becoming Dick Richard Breggs
Freddy Got Fingered Darren Gary & Mike Mike Bonner ( voice ) 2002 Back by Midnight Sheriff Hubbard Sorority Boys Doofer / Roberta 2003 Family Tree Jake Kart
Racer Zee 2005 Lucky 13 Bleckman Because of Winn - Dixie Policeman Robots Lugnut ( voice ) 2006 Employee of the Month Russell Porpis - Gunders Surf School Rip
Ergo Proxy Head Dog Cop ( voice ) 1 episode 2007 Meet the Robinsons Carl ( voice ) The Dukes of Hazzard : The Beginning Sheriff Rosco P. Coltrane 2008 Bachelor
Party 2 : The Last Temptation Derek Madagascar : Escape 2 Africa Giraffe ( voice ) 2009 Fudgy Wudgy Fudge Face Elmore P.
</item 5>

<item 6>
The Dukes of Hazzard (film)
They are later arrested by Atlanta Police after running from campus police . Back in Hazzard , Daisy learns , with the help of Sheriff 's Deputy Enos Strate (
Michael Weston ) , that Billy Prickett has been hired by Boss Hogg to participate in the rally as a ringer . Boss Hogg then heads to Atlanta , where he informs
the Duke boys , in lock - up , that they are too late to stop him and reveals that the vote on Hogg 's proposition is at the same time as the rally , explaining
Billy Prickett 's involvement . During a transfer from detainment , Daisy helps the boys escape from the patrol car , and they speed home to try to inform the
townsfolk , escaping Atlanta Police , and the Georgia State Patrol after Bo outmaneuvers the city cops . Upon returning home , the Dukes discover that Boss Hogg
and Rosco had taken Uncle Jesse and Pauline hostage , an obvious trap for the boys , and that Billy is in on the scheme because he is ashamed of the town 's low
status .
</item 6>

<item 7>
The Dukes of Hazzard (film)
The cousins ' primary mode of transportation is an orange 1969 Dodge Charger that the boys affectionately refer to as the `` General Lee '' . Along the way ,
the family is tormented by corrupt Hazzard County Commissioner Jefferson Davis Hogg , widely known as `` Boss Hogg '' ( Burt Reynolds ) , and his willing but
dimwitted henchman , Sheriff Rosco P. Coltrane ( M.C. Gainey ) . After Rosco has the General Lee impounded after Bo and Luke 's attempt to run away from a
daughter of one their many moonshine customers , Billy Prickett ( James Roday ) , a famous stock - car driver , enters Hazzard to participate in the rally .
Meanwhile , Rosco plants a fake moonshine still ( `` ' cause he 's too dumb to find the real one '' ) in Uncle Jesse 's barn and seizes the Duke property in the
interest of eminent domain for Boss Hogg , forcing the family to temporarily reside with neighbor and Uncle Jesse 's love - interest , Pauline ( Lynda Carter )
.
</item 7>

<item 8>
California Highway Patrol
They spotted the car , fell in behind , called for backup , and began the stop procedure . When the suspects ' vehicle came to a halt in the parking lot , the
driver was instructed to step out of the vehicle and spread his hands on the hood . Gore approached him and Frago moved to the passenger side . The passenger
side door suddenly swung open and the passenger sprung out , firing at Frago , who fell with two shots in his chest . The gunman , later identified as Jack
Twinning , then turned and fired once at Gore , who returned fire . In that moment the driver , Bobby Davis , turned and shot Gore twice at close range . Both
officers died instantly . When Officers James Pence and George Alleyn drove in moments later , they could not see suspects or other officers , but both
immediately came under fire . Pence put out an 11 ‐ 99 call ( `` officer down '' ) then took cover behind the passenger door . Alleyn grabbed his shotgun , and
positioned himself behind the driver - side door . Both officers were mortally wounded in the ensuing exchange , and one suspect was hit .
</item 8>

<item 9>
The Dukes of Hazzard (film)
The characters are now campus security officers , who give a warning to the Duke boys for driving too slowly . Paul Soter appears as TV newsman Rick Shakely ,
who is reporting from the Hazzard Road Rally . Charlie Finn , who played a dimwitted fast - food employee in Super Troopers , appears as a dimwitted geology
student who assists the Duke boys with the coal samples . Other notable cameos During the bar - fight scene , Indy - car driver A.J. Foyt IV appears as himself
. Bloopers roll under the ending credits , one of which features Rip Taylor interrupting the bedroom scene with Luke and the two college girls . Taylor had
previously appeared with Knoxville in Jackass : The Movie . Production ( edit ) Stunts ( edit ) Knoxville said he was initially reluctant to take on the role ,
but was persuaded by script changes and the presence of Dan Bradley as stunt coordinator and second unit for the car chase scenes . Knoxville praised him ,
saying , `` everyone in Hollywood wants Dan Bradley to shoot their car stuff '' . Locations ( edit ) The majority of the film was shot in and around Clinton ,
Louisiana .
</item 9>

<item 10>
A California Highway Patrol officer, fearing police would mistake him for a gunman, hid his service revolver and directed them toward the suspects. Jack
Twinning and Bobby Davis escaped and split up. After a nine-hour search, Twinning took a man hostage, then killed himself with a shotgun stolen from the officer
when police stormed the house. Davis was captured and convicted of four murders. His death sentence was commuted to life after California courts ruled the death
penalty unconstitutional.
</item 10>
=== END ITEM id=5b0ed58ebe4b4612 ===

=== ITEM id=01df23d6140b7f4d ===
Gold sentences and facts:
Sentence 0: A justice of the peace ( JP ) is a judicial officer , of a lower or puisne court , elected or appointed by means of a commission ( letters patent ) to keep the peace . In past centuries the term commissioner of the peace was often used with the same meaning . Depending on the jurisdiction , such justices dispense summary justice or merely deal with local administrative applications in common law jurisdictions . Justices of the peace are appointed or elected from the citizens of the jurisdiction in which they serve , and are ( or were ) usually not required to have any formal legal education in order to qualify for the office . Some jurisdictions have varying forms of training for JPs .
  Fact 0.0: A justice of the peace (JP) is a judicial officer.
  Fact 0.1: A justice of the peace is a judicial officer of a lower or puisne court.
  Fact 0.2: A justice of the peace is elected or appointed by means of a commission (letters patent).
  Fact 0.3: A justice of the peace is elected or appointed to keep the peace.
  Fact 0.4: In past centuries the term commissioner of the peace was often used with the same meaning as justice of the peace.
  Fact 0.5: Depending on the jurisdiction, justices of the peace dispense summary justice.
  Fact 0.6: Depending on the jurisdiction, justices of the peace merely deal with local administrative applications in common law jurisdictions.
  Fact 0.7: Justices of the peace are appointed or elected from the citizens of the jurisdiction in which they serve.
  Fact 0.8: Justices of the peace are (or were) usually not required to have any formal legal education in order to qualify for the office.
  Fact 0.9: Some jurisdictions have varying forms of training for justices of the peace.

Retrieved items:
<item 1>
Justice of the peace
JPs , who are also volunteers , are selected through an extensive interview , written exam and practical testing . They are recommended by the state attorney -
general and appointed by the governor - in - council , and it is their job to authorise and witness statutory declarations and affidavits within the state of
Victoria . There are currently around 4,800 JPs serving in all areas of the state . Bail justices ( edit ) The role of a bail justice is to hear bail
applications ( under the Bail Act 1977 ) and to hear applications for interim accommodation orders for children ( under the Children and Young Persons Act 1989
) within Victoria . Bail justices , once appointed , may remain in their role until they turn 70 years of age ( although they must be under 65 at the time of
their appointment ) . They are often required to attend call outs and rule on bail applications or protection applications for children in danger on weekends
and late at night when the courts are closed , but they can also witness Victorian statutory declarations and affidavits . Candidates must successfully complete
a three - day training course run by the Department of Justice .
</item 1>

<item 2>
Justice of the peace
These courts were introduced in 2009 as a replacement for the district courts ( established in 1975 ) , which in turn replaced burgh police courts . Justices
sit alone or in threes with a qualified legal assessor as convener or clerk of court . They handle many cases of breaches of the peace , drunkenness , minor
assaults , petty theft , and offences under the Civic Government ( Scotland ) Act 1982 . The maximum sentencing power of a justice of the peace is 60 days
imprisonment , or a fine up to £ 2,500 , or both , and the ability to disqualify drivers . In 2006 , the Scottish Government announced its intention to unify
the management of the sheriff and district courts in Scotland , but retaining lay justices , as part of its initiative to create a unified judiciary under the
Lord President . Following the passage of the Criminal Proceedings etc . ( Reform ) ( Scotland ) Act 2007 the justice of the peace courts were implemented on a
sheriffdom - by - sheriffdom basis . In Glasgow , the volume of business required the employment of three solicitors as `` stipendiary magistrates '' who sat in
place of the lay justices .
</item 2>

<item 3>
Justice of the peace
JPs are chosen under the Governor - General 's discretion . Malaysia ( edit ) In Malaysia , justices of the peace have largely been replaced in magistrates '
courts by legally qualified ( first - class ) stipendiary magistrates . However , state governments continue to appoint justices of the peace as honours . In
2004 , some associations of justices of the peace pressed the federal government to allow justices of the peace to sit as second - class magistrates in order to
reduce the backlog of cases in the courts . New Zealand ( edit ) A Justice of the Peace in New Zealand is someone of good stature in the community who is
authorized to witness and sign statutory declarations and affidavits as well as certify documents . They may also perform citizenship ceremonies and act as a
Visiting Justice in prisons the function of which is to record a prisoner 's grievance and decide on its validity . They have certain powers to issue search
warrants , and ( in conjunction with another justice of the peace ) may try minor criminal trials in the district court and exercise powers to remand defendants
in custody , grant bail , and adjourn court hearings .
</item 3>

<item 4>
Justice of the peace
Justice of the peace - wikipedia Justice of the peace Jump to : navigation , search This article needs additional citations for verification . Please help
improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( January 2017 ) ( Learn how and when to
remove this template message ) Taos County , New Mexico . Justice of the Peace Montoyo hears a minor case . ( 1941 ) A justice of the peace ( JP ) is a judicial
officer , of a lower or puisne court , elected or appointed by means of a commission ( letters patent ) to keep the peace . In past centuries the term
commissioner of the peace was often used with the same meaning . Depending on the jurisdiction , such justices dispense summary justice or merely deal with
local administrative applications in common law jurisdictions . Justices of the peace are appointed or elected from the citizens of the jurisdiction in which
they serve , and are ( or were ) usually not required to have any formal legal education in order to qualify for the office . Some jurisdictions have varying
forms of training for JPs . Contents ( hide ) 1 History 2 Modern use 2.1 Australia 2.1.
</item 4>

<item 5>
Justice of the peace
Administrative justice of the peace are court officers appointed by the Minister of Justice , and perform duties such as receiving criminal informations and
issuing warrants . Presiding justice of the peace are appointed by commission under the Great Seal , and can try some criminal matters and issue warrants . They
are appointed from advocates of at least ten years ' standing and serve full - time until the age of 70 . Yukon ( edit ) In Yukon , justices of the peace are
lay officers of the court . They sit in the Justice of the Peace Court , which is part of the Territorial Court of Yukon . Hong Kong ( edit ) In Hong Kong , the
historical functions of justices of the peace have been replaced by full - time , legally qualified magistrates . Nowadays , justices of the peace are
essentially titles of honour given by the Government to community leaders , and to certain officials while they are in their terms of offices . They have no
judicial functions , and their main duties include visiting prisons , institutions for young offenders and drug addicts , psychiatric hospitals , remand homes ,
places of refuge , reception and detention centres , administering statutory declarations , and serving as members of advisory panels .
</item 5>

<item 6>
Justice of the peace
Their judicial powers were replaced by full - time , legally qualified district justices ( now called district judges ) and their quasi-judicial powers by lay
Peace Commissioners . Peace commissioners may sign statutory declarations , and may issue summons and search warrants to the Garda Síochána ( Irish police ) . A
Peace Commissioner can not sign an Affidavit , in Ireland that function is performed by a Commissioner for Oaths . Jamaica ( edit ) A justice of the peace ( JP
) , according to the Ministry of Justice , is a person of unquestionable integrity who seeks to promote and protect the rights of the individual and helps to
provide justice to persons in a particular community . Additionally , the JP serves as a justice in petty court sessions , attends juvenile court sessions ,
issues summonses , considers applications for bail , explains and signs legal documents , sits on licensing panels , and gives counsel / advice . Any Jamaican
citizen that can speak and write English is eligible to become a JP . Any club / organisation / citizen can recommend someone to become JP for a community . JPs
are chosen under the Governor - General 's discretion .
</item 6>

<item 7>
Justice of the peace
Justices of the peace are elected every two years to these partisan offices . Massachusetts , Connecticut , Vermont ( edit ) Justices of the peace in the
Commonwealth of Massachusetts are often called on to perform marriages and , especially same - sex marriages , which certain religious officials are not willing
to oversee . Justices of the peace in Connecticut can preside over same - sex marriages . Unlike Massachusetts , Connecticut JPs are not penalized for refusing
to perform such ceremonies . Justices of the peace in Connecticut , Massachusetts , and Vermont have the same general oath - giving powers as a notary public .
Louisiana Are elected to serve six year terms as the judicial authority of a ward or district . Have jurisdiction in civil matters when the amount in dispute
does not exceed $5,000 . Do not have jurisdiction when a title to real estate is involved , when the state or any political subdivision is a defendant , or in
successions or probate matters . Can perform marriage ceremonies Minnesota ( edit ) In Minnesota , the history of justices of the peace dates back hundreds of
years to a time when they were considered to be some of the most powerful officials in terms of the authority that they had to influence change in a community .
</item 7>

<item 8>
Justice of the peace
United states ( edit ) In some US states , the justice of the peace is a judge of a court of limited jurisdiction , a magistrate , or a quasi-judicial official
with certain statutory or common law magisterial powers . Some states have special qualifications or unique features for the office . The justice of the peace
typically presides over a court that hears misdemeanor cases , traffic violations , and other petty criminal infractions . The justice of the peace may also
have authority over cases involving small debts , landlord and tenant disputes , or other small claims court proceedings . Proceedings before justices of the
peace are often faster and less formal than the proceedings in other courts . In some jurisdictions a party convicted or found liable before a justice of the
peace may have the right to a trial de novo before the judge of a higher court rather than an appeal strictly considered . A justice of the peace also performs
civil marriages . Arizona ( edit ) A justice of the peace has the same jurisdiction as a municipal magistrate with respect to traffic and misdemeanor cases and
restraining orders , though over cases whose affairs are not contained within the confines of a single municipality .
</item 8>

<item 9>
Justice of the peace
A JP has contempt power of $100 and up to 3 days in jail per occurrence . A JP is also authorized to perform marriage ceremonies . Replacement with other courts
( edit ) In many states , the office of justice of the peace has been abolished or transferred to another court , such as the magistrate court . In larger
cities , cases may be heard in a municipal court which has jurisdiction only within that city . Most efforts to abolish the office of justice of the peace have
been led by the American Bar Association , which views non-lawyer judges as no longer necessary , as there are now far more persons with formal legal education
than in the past when justices of the peace were first used . California formerly had justice of the peace courts staffed by lay judges , but began phasing them
out after a landmark 1974 decision in which the Supreme Court of California unanimously held that it was a violation of federal due process ( under the
Fourteenth Amendment to the U.S. Constitution ) to allow a non-lawyer to preside over a criminal trial which could result in incarceration of the defendant .
</item 9>

<item 10>
Justice of the peace
Additionally , the Justice Court hears cases involving county ordinances ( ordinances enacted by the board of supervisors that apply only to unincorporated
areas ) , civil lawsuits up to a limit of $10,000 , small claims cases up to $2,500 , and issues evictions , called writs of restitution ( after a forcible
detainer or special detainer action ( eviction ) being successfully completed by a landlord ) . Justices of the peace , also called JPs , or Judges of the
Justice Court , are elected in partisan elections for four - year terms from specific districts called precincts . They have the same authority and
responsibility as all other judges in the state with respect to performing marriages , administrating oaths , adhering to the code of judicial conduct , and all
aspects of justice administration . However , Arizona law does not require justices of the peace to be lawyers . Many justices of the peace are not legally
trained , although all are required by the Arizona Supreme Court to complete a course at the Arizona Judicial College . As with JPs , municipal judges in
Arizona are not required to be lawyers . Arkansas ( edit ) This section does not cite any sources .
</item 10>
=== END ITEM id=01df23d6140b7f4d ===

=== ITEM id=431bec3a1f8350c0 ===
Gold sentences and facts:
Sentence 0: The time that has elapsed since the Camp David Accords has left no doubt as to their enormous ramifications on Middle Eastern politics . Most notably , the perception of Egypt within the Arab world changed . With the most powerful of the Arab militaries and a history of leadership in the Arab world under Nasser , Egypt had more leverage than any of the other Arab states to advance Arab interests . Egypt was subsequently suspended from the Arab League from 1979 until 1989 .
  Fact 0.0: Time has elapsed since the Camp David Accords.
  Fact 0.1: The passage of time has left no doubt about the enormous ramifications of the Camp David Accords on Middle Eastern politics.
  Fact 0.2: The perception of Egypt within the Arab world changed.
  Fact 0.3: The change in perception of Egypt was the most notable effect.
  Fact 0.4: Egypt had the most powerful of the Arab militaries.
  Fact 0.5: Egypt had a history of leadership in the Arab world under Nasser.
  Fact 0.6: Egypt had more leverage than any other Arab state to advance Arab interests.
  Fact 0.7: Egypt was subsequently suspended from the Arab League.
  Fact 0.8: Egypt's suspension from the Arab League lasted from 1979 until 1989.

Retrieved items:
<item 1>
Camp David Accords
Lastly , the biggest consequence of all may be in the psychology of the participants of the Arab - Israeli conflict . The success of Begin , Sadat , and Carter
at Camp David demonstrated to other Arab states and entities that negotiations with Israel were possible -- that progress results only from sustained efforts at
communication and cooperation . Despite the disappointing conclusion of the 1993 Oslo Accords between the PLO and Israel , and even though the 1994 Israel --
Jordan peace treaty has not fully normalized relations with Israel , both of these significant developments had little chance of occurring without the precedent
set by Camp David . Public support Although most Israelis supported the Accords , the Israeli settler movement opposed them because Sadat would not agree to a
treaty in which Israel had any presence in the Sinai Peninsula at all , Israel had to withdraw from the entire Sinai Peninsula . Israeli settlers tried to
prevent the government from dismantling their settlements . In Israel , there is lasting support of the Camp David Peace Accords , which have become a national
consensus , supported by 85 % of Israelis according to a 2001 poll taken by the Jaffee Center for Strategic Studies ( Israel based ) .
</item 1>

<item 2>
Camp David Accords
The efforts initially focused on a comprehensive resolution of disputes between Israel and the Arab countries , gradually evolving into a search for a bilateral
agreement between Israel and Egypt . Upon assuming office on January 20 , 1977 , President Carter moved to rejuvenate the Middle East peace process that had
stalled throughout the 1976 presidential campaign in the United States . Following the advice of a Brookings Institution report , Carter opted to replace the
incremental , bilateral peace talks which had characterized Henry Kissinger 's shuttle diplomacy following the 1973 Yom Kippur War with a comprehensive ,
multilateral approach . The Yom Kippur War further complicated efforts to achieve the objectives written in United Nations Security Council Resolution 242 .
Israel 's Prime Minister Yitzhak Rabin and his successor , Menachem Begin , were both skeptical of an international conference . While Begin , who took office
in May 1977 , officially favored the reconvening of the conference , perhaps even more vocally than Rabin , and even accepted the Palestinian presence , in
actuality the Israelis and the Egyptians were secretly formulating a framework for bilateral talks .
</item 2>

<item 3>
Camp David Accords
^ Jump up to : Stein , 1999 , p. 254 . Jump up ^ Gold , 175 Jump up ^ UNGA , 7 December 1978 , Resolution 33 / 28 A. Question of Palestine ( doc.nr . A / RES /
33 / 28 ) Jump up ^ UNGA , 6 December 1979 , Resolution 34 / 70 . The situation in the Middle East ( doc.nr . A / RES / 34 / 70 ) Jump up ^ UNGA , 12 December
1979 , Resolution 34 / 65 B. Question of Palestine Archived 29 March 2012 at the Wayback Machine ... ( doc.nr . A / RES / 34 / 65 ( A-D ) ) Jump up ^ Jimmy
Carter Library , Framework for the Conclusion of a Peace Treaty between Egypt and Israel Archived 16 February 2013 at the Wayback Machine . Jump up ^ `` Egypt
'' Archived 26 January 2011 at the Wayback Machine . U.S. Department of State . March 2008 . 28 April 2008 .
</item 3>

<item 4>
Camp David Accords
Accompanied by their capable negotiating teams and with their respective interests in mind , both leaders converged on Camp David for 13 days of tense and
dramatic negotiations from 5 to 17 September 1978 . By all accounts , Carter 's relentless drive to achieve peace and his reluctance to allow the two men to
leave without reaching an agreement are what played the decisive role in the success of the talks . Carter 's advisers insisted on the establishment of an
Egyptian - Israeli agreement which would lead to an eventual solution to the Palestine issue . They believed in a short , loose , and overt linkage between the
two countries amplified by the establishment of a coherent basis for a settlement . However , Carter felt they were not `` aiming high enough '' and was
interested in the establishment of a written `` land for peace '' agreement with Israel returning the Sinai Peninsula and West Bank . Numerous times both the
Egyptian and Israeli leaders wanted to scrap negotiations , only to be lured back into the process by personal appeals from Carter .
</item 4>

<item 5>
Camp David Accords
1 Key points of the West Bank and Gaza section 3.1. 2 UN Rejection of the Middle East Framework 3.2 Framework Peace Treaty Egypt and Israel 4 Consequences 5
Public support 6 Criticism of the Accords 7 Assassination of Anwar Sadat 8 Arab -- Israeli peace diplomacy and treaties 9 See also 10 References 11 Further
reading 12 External links Preceding diplomacy Carter initiative Carter 's and Secretary of State Cyrus Vance 's exploratory meetings gave a basic plan for
reinvigorating the peace process based on a Geneva Peace Conference and had presented three main objectives for Arab -- Israeli peace : Arab recognition of
Israel 's right to exist in peace , Israel 's withdrawal from occupied territories gained in the Six - Day War through negotiating efforts with neighboring Arab
nations to ensure that Israel 's security would not be threatened and securing an undivided Jerusalem . The Camp David Accords were the result of 14 months of
diplomatic efforts by Egypt , Israel , and the United States that began after Jimmy Carter became President .
</item 5>

<item 6>
Camp David Accords
Arab -- Israeli peace diplomacy and treaties Treaties and meetings Paris Peace Conference , 1919 Faisal -- Weizmann Agreement ( 1919 ) 1949 Armistice Agreements
Geneva Conference ( 1973 ) Camp David Accords ( 1978 ) Egypt -- Israel Peace Treaty ( 1979 ) Madrid Conference of 1991 Oslo Accords ( 1993 ) Israel -- Jordan
peace treaty ( 1994 ) Camp David 2000 Summit General articles International law and the Arab -- Israeli conflict Israeli -- Palestinian peace process List of
Middle East peace proposals Projects working for peace among Israelis and Arabs See also 1948 Arab -- Israeli War 1956 Suez War 1967 Six - Day War 1970 War of
Attrition 1973 Yom Kippur War Arab -- Israeli conflict Arab League and the Arab -- Israeli conflict Egypt -- Israel relations Israeli -- Palestinian conflict
Proposals for a Palestinian state References Jump up ^ Camp David Accords -- Israeli Ministry of Foreign Affairs Archived 3 September 2011 at the Wayback
Machine . Jump up ^ http://www.cartercenter.org/news/documents/doc1482.html ^ Jump up to : Stein , Kenneth .
</item 6>

<item 7>
Camp David Accords
Camp David Accords - wikipedia Camp David Accords Jump to : navigation , search Camp David Accords Framework for Peace in the Middle East and Framework for the
Conclusion of a Peace Treaty between Egypt and Israel Celebrating the signing of the Camp David Accords : Menachem Begin , Jimmy Carter , Anwar Sadat Type
Bilateral treaty Signed 17 September 1978 ( 1978 - 09 - 17 ) Location Washington , DC , United States Original signatories Egypt Israel Signatories Menachem
Begin ( Prime Minister of Israel ) Anwar Sadat ( President of Egypt ) Jimmy Carter ( President of the United States ) Ratifiers Egypt Israel Language English
French The Camp David Accords were signed by Egyptian President Anwar Sadat and Israeli Prime Minister Menachem Begin on 17 September 1978 , following twelve
days of secret negotiations at Camp David . The two framework agreements were signed at the White House , and were witnessed by United States President Jimmy
Carter . The second of these frameworks ( A Framework for the Conclusion of a Peace Treaty between Egypt and Israel ) led directly to the 1979 Egypt -- Israel
Peace Treaty .
</item 7>

<item 8>
Camp David Accords
Meir Rosenne and Aharon Barak . Furthermore , the absence of the media contributed to the Accord 's successes : there were no possibilities provided to either
leader to reassure his political body or be driven to conclusions by members of his opposition . An eventual scrap of negotiations by either leader would have
proven disastrous , resulting in taking the blame for the summit 's failure as well as a disassociation from the White House . Ultimately , neither Begin nor
Sadat was willing to risk those eventualities . Both of them had invested enormous amounts of political capital and time to reach an agreement . Partial
agreements Egyptian President Anwar Sadat and Israeli Prime Minister Menachem Begin acknowledge applause during a joint session of Congress in Washington , D.C.
, during which President Jimmy Carter announced the results of the Camp David Accords , 18 September 1978 . `` Remarks on the Signing of the Camp David Accords
'' Play media Jimmy Carter , seated with Egyptian President Anwar Sadat and Israeli Prime Minister Menachem Begin , makes statements at a Joint session of the
United States Congress following the Camp David Accords . Problems playing this file ? See media help .
</item 8>

<item 9>
Camp David Accords
According to The Continuum Political Encyclopedia of the Middle East : The normalization of relations ( between Israel and Egypt ) went into effect in January
1980 . Ambassadors were exchanged in February . The boycott laws were repealed by Egypt 's National Assembly the same month , and some trade began to develop ,
albeit less than Israel had hoped for . In March 1980 regular airline flights were inaugurated . Egypt also began supplying Israel with crude oil `` . According
to Kenneth Stein in Heroic Diplomacy : Sadat , Kissinger , Carter , Begin , and the Quest for Arab -- Israeli Peace : The Accords were another interim agreement
or step , but negotiations that flowed from the Accords slowed for several reasons . These included an inability to bring the Jordanians into the discussions ;
the controversy over settlements ; the inconclusive nature of the subsequent autonomy talks ; domestic opposition sustained by both Begin and Sadat and , in
Sadat 's case , ostracism and anger from the Arab world ; the emergence of a what became a cold peace between Egypt and Israel ; and changes in foreign policy
priorities including discontinuity in personnel committed to sustaining the negotiating process ( . )
</item 9>

<item 10>
Camp David Accords
Problems playing this file ? See media help . The Camp David Accords comprise two separate agreements : `` A Framework for Peace in the Middle East '' and `` A
Framework for the Conclusion of a Peace Treaty between Egypt and Israel '' , the second leading towards the Israel - Egypt Peace Treaty signed in March 1979 .
The agreements and the peace treaty were both accompanied by `` side - letters '' of understanding between Egypt and the U.S. and Israel and the U.S. Framework
for peace in the Middle East The preamble of the `` Framework for Peace in the Middle East '' starts with the basis of a peaceful settlement of the Arab --
Israeli conflict : The agreed basis for a peaceful settlement of the conflict between Israel and its neighbors is United Nations Security Council Resolution 242
, in all its parts . The framework itself consists of 3 parts . The first part of the framework was to establish an autonomous self - governing authority in the
West Bank and the Gaza strip and to fully implement Resolution 242 . The Accords recognized the `` legitimate rights of the Palestinian people '' , a process
was to be implemented guaranteeing the full autonomy of the people within a period of five years .
</item 10>
=== END ITEM id=431bec3a1f8350c0 ===
