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

=== ITEM id=db1641d0a32a94e9 ===
Gold sentences and facts:
Sentence 0: Bruce Willis as John McClane Jai Courtney as John `` Jack '' McClane , Jr . Sebastian Koch as Yuri Komarov Yuliya Snigir as Irina Komarova Sergei Kolesnikov as Viktor Chagarin Radivoje Bukvić as Alik Mary Elizabeth Winstead as Lucy McClane
  Fact 0.0: Bruce Willis plays John McClane.
  Fact 0.1: Jai Courtney plays John 'Jack' McClane, Jr.
  Fact 0.2: Sebastian Koch plays Yuri Komarov.
  Fact 0.3: Yuliya Snigir plays Irina Komarova.
  Fact 0.4: Sergei Kolesnikov plays Viktor Chagarin.
  Fact 0.5: Radivoje Bukvić plays Alik.
  Fact 0.6: Mary Elizabeth Winstead plays Lucy McClane.

Retrieved items:
<item 1>
A Good Day to Die Hard
Casting ( edit ) Bruce Willis returned as John McClane , and has expressed a desire to shoot A Good Day to Die Hard and a sixth installment in the series before
retiring the character . When casting the role of Jack McClane , the studios considered several actors , including Liam Hemsworth and James Badge Dale , before
ultimately settling on Australian actor Jai Courtney . Mary Elizabeth Winstead also appeared in the film , reprising her role as McClane 's daughter Lucy .
However all her scenes were cut from the Extended release . Sebastian Koch played the film 's primary antagonist , Yuri Komarov , while Yuliya Snigir and Cole
Hauser featured as secondary characters Irina and Collins . The cast was completed by actors Amaury Nolasco as a friend of McClane , Pasha D. Lychnikoff as a
taxi driver , and Megalyn Echikunwoke , Anne Vyalitsyna , and Ivan Kamaras in smaller roles . Filming ( edit ) Production began in Hungary in April 2012 , with
the capital Budapest standing in for Moscow .
</item 1>

<item 2>
A Good Day to Die Hard
Komarov remarks that Jack will get to watch his father die , prompting Jack to hurl him off the rooftop into the path of the spinning helicopter rotors ,
killing him . As Jack and John reunite inside the building , Irina tries to avenge her father by ramming the helicopter , now out of ammunition , into the
building in a suicide attack . Father and son survive by leaping off the building into a large pool of water , as the helicopter crashes and explodes , killing
Irina . In the end , the McClanes return home and reunite with Lucy on the tarmac . Cast ( edit ) Bruce Willis as John McClane Jai Courtney as John `` Jack ''
McClane , Jr . Sebastian Koch as Yuri Komarov Yuliya Snigir as Irina Komarova Sergei Kolesnikov as Viktor Chagarin Radivoje Bukvić as Alik Mary Elizabeth
Winstead as Lucy McClane Additional cast members include CIA operatives : Aldis Hodge as Lt. Foxy and Cole Hauser as Mike Collins . Amaury Nolasco appears as
Murphy , an NYPD detective and McClane 's friend .
</item 2>

<item 3>
A Good Day to Die Hard
Roman Luknár makes an apperance as Anton , a business associate of Chagarin . Ganxsta Zolee makes a brief cameo as a MRAP driver . Production ( edit )
Pre-production ( edit ) Production was formally announced in 2010 , with X-Men Origins : Wolverine and The A-Team writer Skip Woods confirmed as the film 's
screenwriter . Noam Murro was originally attached to direct the film but left production to direct the 300 prequel , 300 : Rise of an Empire . John Moore was
subsequently drafted in to replace him . The film was originally titled `` Die Hard 24 / 7 '' . The media speculated that the film would be a crossover between
the Die Hard and 24 series , with Kiefer Sutherland to reprise his role as Jack Bauer alongside John McClane . This was never confirmed by the studio , and the
film 's title was later revealed to be A Good Day to Die Hard -- with no further mention of any involvement from the 24 series -- with a release date of
February 14 , 2013 .
</item 3>

<item 4>
A Good Day to Die Hard
A Good Day to Die Hard was a box office success but received overwhelmingly negative reviews for its implausible action sequences , cinematography , weak plot ,
cliched script , and lack of characterization , although the special effects were praised . It is the first film produced by TSG Entertainment , since
distributor 20th Century Fox 's departure from Dune Entertainment upon the completion of their distribution contract at the end of 2012 . Contents ( hide ) 1
Plot 2 Cast 3 Production 3.1 Pre-production 3.2 Casting 3.3 Filming 3.4 Post-production 4 Soundtrack 5 Release 6 Reception 6.1 Box office 6.2 Home media 7
References 8 External links Plot ( edit ) In Moscow , Viktor Chagarin , a high - ranking , but corrupt , Russian official plans to incriminate former
billionaire and government whistleblower Yuri Komarov in an imminent rigged trial unless Komarov hands over a secret file believed to contain evidence
incriminating Chagarin . Separately , Jack McClane , who has been arrested after an assassination attempt , negotiates for a shorter sentence by offering to
testify against Komarov .
</item 4>

<item 5>
A Good Day to Die Hard
March 21 , 2013 . External links ( edit ) Official website A Good Day to Die Hard on IMDb A Good Day to Die Hard at Box Office Mojo A Good Day to Die Hard at
Rotten Tomatoes A Good Day to Die Hard at Metacritic Die Hard Films Die Hard Die Hard 2 Die Hard with a Vengeance Live Free or Die Hard A Good Day to Die Hard
Characters John McClane Video games Die Hard Die Hard Arcade Die Hard Trilogy Die Hard Trilogy 2 : Viva Las Vegas Die Hard : Nakatomi Plaza Die Hard : Vendetta
Other Die Hard : Year One The Detective novel film Nothing Lasts Forever 58 Minutes Dynamite Cop Val Verde Roderick Thorp Films directed by John Moore Behind
Enemy Lines ( 2001 ) Flight of the Phoenix ( 2004 ) The Omen ( 2006 ) Max Payne ( 2008 ) A Good Day to Die Hard ( 2013 ) I.T.
</item 5>

<item 6>
A Good Day to Die Hard
Rick Groen of The Globe and Mail criticized the action scenes as being `` messy '' , but concluded his review saying that the film `` continues the franchise
without undue embarrassment . '' Peter Howell of The Toronto Star remarked that Willis and Courtney made a strong estranged family duo and that the film had a
nice drinking game routine going for it with how many times McClane exclaims `` I 'm on vacation ! '' during the running time . Box Office ( edit ) A Good Day
to Die Hard grossed $67,349,198 in North America and $237,304,984 in other territories for a worldwide total of $304,654,182 , collecting more than three times
its $92 million budget . In North America , A Good Day to Die Hard grossed an estimated $840,000 from its night showings at 2,328 locations on Wednesday ,
February 13 , 2013 . The next day , at an additional 1,225 locations , the film managed to accumulate $8,239,116 , opening at # 2 behind Safe Haven .
</item 6>

<item 7>
A Good Day to Die Hard
Retrieved 2012 - 02 - 22 . Jump up ^ `` Mary Elizabeth Winstead to Return as John McClane 's Daughter in A Good Day To Die Hard '' . Collider.com. 2012 - 08 -
07 . Retrieved 2012 - 08 - 07 . Jump up ^ http://www.movie-censorship.com/report.php?ID=175700 ^ Jump up to : `` A Good Day to Die Hard ' Finds a Couple of Male
and Female Villains '' . firstshowing.net. 2012 - 04 - 04 . Retrieved 2012 - 04 - 04 . Jump up ^ `` Cole Hauser Joins Bruce Willis In A Good Day To Die Hard ''
. deadline.com. 2012 - 05 - 08 . Retrieved 2012 - 05 - 08 . ^ Jump up to : Sneider , Jeff ( 2012 - 05 - 14 ) . `` Latest in ' Die Hard ' series adds two '' .
Variety . Retrieved 2012 - 05 - 31 .
</item 7>

<item 8>
A Good Day to Die Hard
A Good Day to Die Hard - wikipedia A Good Day to Die Hard Jump to : navigation , search A Good Day to Die Hard Theatrical release poster Directed by John Moore
Produced by Alex Young Wyck Godfrey Written by Skip Woods Based on Characters by Roderick Thorp Starring Bruce Willis Jai Courtney Sebastian Koch Rasha Bukvić
Cole Hauser Yuliya Snigir Music by Marco Beltrami Cinematography Jonathan Sela Edited by Dan Zimmerman Production company Giant Pictures TSG Entertainment
Distributed by 20th Century Fox Release date January 31 , 2013 ( 2013 - 01 - 31 ) ( Leicester Square ) February 14 , 2013 ( 2013 - 02 - 14 ) ( United States )
Running time 97 minutes Country United States Language English Russian Budget $92 million Box office $304.7 million A Good Day to Die Hard is a 2013 American
action thriller film and the fifth installment in the Die Hard film series . The film was directed by John Moore and written by Skip Woods , and stars Bruce
Willis as John McClane .
</item 8>

<item 9>
A Good Day to Die Hard
`` Get to the Choppa ! '' 2 : 59 23 . `` Chopper Takedown '' 3 : 26 24 . `` It 's Hard to Kill a McClane '' 2 : 59 25 . `` Triple Vodka Rhapsody '' 1 : 55 26 .
`` McClane 's Brain '' 2 : 00 Release ( edit ) # 46 TruSpeed Porsche GT3 during a race of the 2012 Supercup ( that ran as a support to the 2012 USGP ) promoting
the release date on February 14 , 2013 . On January 31 , 2013 , 20th Century Fox held a special tribute to the 25th anniversary of the Die Hard series by
unveiling a mural of a scene from Die Hard ( 1988 ) on Sound Stage 8 of the Fox Lot . Afterward , the premiere screening of A Good Day to Die Hard was shown .
Two additional premieres were held in Europe leading up to the film 's commercial release : one in Berlin , Germany on February 4 and the second in London ,
England on February 7 .
</item 9>

<item 10>
A Good Day to Die Hard
The main plot finds McClane travelling to Russia to get his estranged son , Jack , an undercover CIA agent , out of prison . He is soon caught in the crossfire
of a global terrorist plot . Alongside Willis , the film also stars Jai Courtney , Cole Hauser , Yuliya Snigir and Sebastian Koch as the film 's villain . Talks
of a fifth Die Hard film began before the release of the fourth installment , Live Free or Die Hard , with Willis affirming that the latter would not be the
last in the series , but pre-production did not start until September 2011 , when John Moore was officially announced as the director . Filming began in April
2012 , primarily in Budapest , Hungary . A Good Day to Die Hard premiered in Los Angeles on January 31 , 2013 , coinciding with the unveiling of a Die Hard
mural at the Fox Lot , and was released in certain East and Southeast Asian countries on February 7 and in the United States and Canada on February 13 . It is
the first Die Hard film to use Dolby Atmos Surround Mixing and the first to also be released in IMAX theaters .
</item 10>
=== END ITEM id=db1641d0a32a94e9 ===

=== ITEM id=d2b24b09b46e9386 ===
Gold sentences and facts:
Sentence 0: The Osborne 1 is considered the first true mobile computer by most historians . Adam Osborne founded Osborne Computer and formed the Osborne 1 in 1981 . The Osborne 1 had a five - inch screen , incorporating a modem port , two 5 1 / 4 floppy drives , a big collection of bundled software applications , and a battery pack . The computer company was a failure and did not last for very long . Although it was large and heavy compared to today 's laptops , with a tiny 5 '' CRT monitor , it had a near - revolutionary impact on business , as professionals were able to take their computer and data with them for the first time . This and other `` luggables '' were inspired by what was probably the first portable computer , the Xerox NoteTaker . The Osborne was about the size of a portable sewing machine , and could be carried on commercial aircraft . The Osborne 1 weighed close to 11 kg and was priced at $1795 .
  Fact 0.0: The Osborne 1 is considered the first true mobile computer by most historians.
  Fact 0.1: Adam Osborne founded Osborne Computer.
  Fact 0.2: Adam Osborne formed the Osborne 1 in 1981.
  Fact 0.3: The Osborne 1 had a five-inch screen.
  Fact 0.4: The Osborne 1 incorporated a modem port.
  Fact 0.5: The Osborne 1 had two 5 1/4 floppy drives.
  Fact 0.6: The Osborne 1 came with a big collection of bundled software applications.
  Fact 0.7: The Osborne 1 had a battery pack.
  Fact 0.8: The Osborne Computer company was a failure.
  Fact 0.9: The Osborne Computer company did not last for very long.
  Fact 0.10: Compared to today's laptops, the Osborne 1 was large and heavy.
  Fact 0.11: The Osborne 1 had a tiny 5-inch CRT monitor.
  Fact 0.12: The Osborne 1 had a near-revolutionary impact on business.
  Fact 0.13: Professionals were able to take their computer and data with them for the first time because of the Osborne 1.
  Fact 0.14: Other 'luggables' were inspired by the Osborne 1.
  Fact 0.15: The Osborne 1 was inspired by what was probably the first portable computer, the Xerox NoteTaker.
  Fact 0.16: The Osborne was about the size of a portable sewing machine.
  Fact 0.17: The Osborne could be carried on commercial aircraft.
  Fact 0.18: The Osborne 1 weighed close to 11 kg.
  Fact 0.19: The Osborne 1 was priced at $1795.
Sentence 1: The portable micro computer the `` Portal '' of the French company R2E Micral CCMC officially appeared in September 1980 at the Sicob show in Paris . The Portal was a portable microcomputer designed and marketed by the studies and developments department of the French firm R2E Micral in 1980 at the request of the company CCMC specializing in payroll and accounting . It was based on an Intel 8085 processor , 8 - bit , clocked at 2 MHz . It was equipped with a central 64K byte Ram , a keyboard with 58 alpha numeric keys and 11 numeric keys ( separate blocks ) , a 32 - character screen , a floppy disk : capacity = 140 000 characters , of a thermal printer : speed = 28 characters / second , an asynchronous channel , a synchronous channel , a 220V power supply . Designed for an operating temperature of 15 ° C to 35 ° C , it weighed 12 kg and its dimensions were 45cm x 45cm x 15cm . It provided total mobility . Its operating system was Prologue .
  Fact 1.0: The portable micro computer the 'Portal' of the French company R2E Micral CCMC officially appeared in September 1980 at the Sicob show in Paris.
  Fact 1.1: The Portal was a portable microcomputer designed and marketed by the studies and developments department of the French firm R2E Micral in 1980.
  Fact 1.2: The Portal was designed at the request of the company CCMC, which specializes in payroll and accounting.
  Fact 1.3: The Portal was based on an Intel 8085 processor.
  Fact 1.4: The Portal's Intel 8085 processor was 8-bit and clocked at 2 MHz.
  Fact 1.5: The Portal was equipped with a central 64K byte RAM.
  Fact 1.6: The Portal had a keyboard with 58 alphanumeric keys and 11 numeric keys in separate blocks.
  Fact 1.7: The Portal had a 32-character screen.
  Fact 1.8: The Portal had a floppy disk with a capacity of 140 000 characters.
  Fact 1.9: The Portal had a thermal printer with a speed of 28 characters per second.
  Fact 1.10: The Portal had an asynchronous channel and a synchronous channel.
  Fact 1.11: The Portal required a 220V power supply.
  Fact 1.12: The Portal was designed for an operating temperature of 15 °C to 35 °C.
  Fact 1.13: The Portal weighed 12 kg.
  Fact 1.14: The Portal's dimensions were 45cm x 45cm x 15cm.
  Fact 1.15: The Portal provided total mobility.
  Fact 1.16: The Portal's operating system was Prologue.
Sentence 2: The first laptop in the modern form was the Grid Compass 1101 , designed by Bill Moggridge in 1979 -- 80 , and released in 1983 . Enclosed in a magnesium case , it introduced the now familiar clamshell design , in which the flat display folded shut against the keyboard . The computer could be run from batteries , and was equipped with a 320 × 200 - pixel electroluminescent display and 384 kilobyte bubble memory . It was not IBM - compatible , and its high price ( US $ 8,000 -- 10,000 ) limited it to specialized applications . However , it was used heavily by the U.S. military , and by NASA on the Space Shuttle during the 1980s . The GRiD 's manufacturer subsequently earned significant returns on its patent rights as its innovations became commonplace . GRiD Systems Corp. was later bought by the Tandy ( now RadioShack ) Corporation .
  Fact 2.0: The first laptop in the modern form was the Grid Compass 1101.
  Fact 2.1: The Grid Compass 1101 was designed by Bill Moggridge in 1979-80.
  Fact 2.2: The Grid Compass 1101 was released in 1983.
  Fact 2.3: The Grid Compass 1101 was enclosed in a magnesium case.
  Fact 2.4: The Grid Compass 1101 introduced the now familiar clamshell design, in which the flat display folded shut against the keyboard.
  Fact 2.5: The Grid Compass 1101 could be run from batteries.
  Fact 2.6: The Grid Compass 1101 had a 320 × 200-pixel electroluminescent display.
  Fact 2.7: The Grid Compass 1101 had 384 kilobyte bubble memory.
  Fact 2.8: The Grid Compass 1101 was not IBM-compatible.
  Fact 2.9: The Grid Compass 1101 had a high price of US $8,000-10,000, which limited it to specialized applications.
  Fact 2.10: The Grid Compass 1101 was used heavily by the U.S. military.
  Fact 2.11: The Grid Compass 1101 was used by NASA on the Space Shuttle during the 1980s.
  Fact 2.12: The GRiD manufacturer subsequently earned significant returns on its patent rights as its innovations became commonplace.
  Fact 2.13: GRiD Systems Corp. was later bought by the Tandy (now RadioShack) Corporation.

Retrieved items:
<item 1>
History of laptops
Another significant machine announced in 1981 , although first sold widely in 1983 , was the Epson HX - 20 . A simple handheld computer , it featured a full -
transit 68 - key keyboard , rechargeable nickel - cadmium batteries , a small ( 120 × 32 - pixel ) dot - matrix LCD with 4 lines of text , 20 characters per
line text mode , a 24 column dot matrix printer , a Microsoft BASIC interpreter , and 16 KB of RAM ( expandable to 32 KB ) . Through at least the late 1980s
Epson continued to release laptops such as the L3s . GRiD Compass ( edit ) Main article : Grid Compass The first laptop in the modern form was the Grid Compass
1101 , designed by Bill Moggridge in 1979 -- 80 , and released in 1983 . Enclosed in a magnesium case , it introduced the now familiar clamshell design , in
which the flat display folded shut against the keyboard . The computer could be run from batteries , and was equipped with a 320 × 200 - pixel
electroluminescent display and 384 kilobyte bubble memory .
</item 1>

<item 2>
History of laptops
Retrieved 11 August 2015 . Jump up ^ See TRS - 80 Model 100 / 102 at old-computers.com Jump up ^ `` Commodore SX - 64 Portable '' . Retrieved 11 August 2015 .
Jump up ^ `` Commodore SX - 64 portable computer '' . Retrieved 11 August 2015 . Jump up ^ `` World 's First Laptop PC '' . Toshiba Science Museum . Retrieved
21 May 2017 . Jump up ^ `` Business Desktops / Calcs '' . HP Computer Museum . Retrieved 7 April 2014 . Jump up ^ Lewis , Peter H. ( 23 October 1988 ) . `` THE
EXECUTIVE COMPUTER ; Compaq Finally Makes a Laptop '' . The New York Times . Jump up ^ Somerson , Paul ( July 1986 ) . `` Is It On Yet ? '' . PC . pp. 122 --
123 . Retrieved 9 January 2015 . Jump up ^ `` IBM Power Systems archived white papers '' . Retrieved 11 August 2015 .
</item 2>

<item 3>
Graph query row:
entity: Grid Compass 1101
relationship: CREATED
related: Bill Moggridge
</item 3>

<item 4>
History of laptops
Cambridge Z88 ( edit ) Another notable computer was the Cambridge Z88 , designed by Clive Sinclair , introduced in 1988 . About the size of an A4 sheet of paper
as well , it ran on standard batteries , and contained basic spreadsheet , word processing , and communications programs . It anticipated the future
miniaturization of the portable computer , and as a ROM - based machine with a small display , can -- like the TRS - 80 Model 100 -- also be seen as a
forerunner of the personal digital assistant . Compaq SLT / 286 ( edit ) By the end of the 1980s , laptop computers were becoming popular among business people
. The COMPAQ SLT / 286 debuted in October 1988 , being the first battery - powered laptop to support an internal hard disk drive and a VGA compatible LCD screen
. It weighed 14 lbs . NEC Ultralite ( edit ) Main article : NEC UltraLite The NEC UltraLite , released in mid-1989 , was perhaps the first `` notebook ''
computer , weighing just 2 kg ( 4.4 lbs ) .
</item 4>

<item 5>
History of laptops
Commodore SX - 64 ( edit ) The Commodore SX - 64 , also known as the Executive 64 , or VIC - 64 in Europe , was a portable , briefcase or suitcase - size ``
luggable '' version of the popular Commodore 64 home computer , and was the first full - color portable computer . The SX - 64 featured a built - in five - inch
composite monitor and a built - in 1541 floppy drive . It weighed 20 pounds . The machine was carried by its sturdy handle , which doubled as an adjustable
stand . It was announced in January 1983 and released a year later , at $995 USD . Kaypro 2000 ( edit ) Possibly the first commercial IBM - compatible laptop
was the Kaypro 2000 , introduced in 1985 . With its brushed aluminum clamshell case , it was remarkably similar in design to modern laptops . It featured a 25
line by 80 character LCD , a detachable keyboard , and a pop - up 90 mm ( 3.5 - inch ) floppy drive .
</item 5>

<item 6>
History of laptops
It was not IBM - compatible , and its high price ( US $ 8,000 -- 10,000 ) limited it to specialized applications . However , it was used heavily by the U.S.
military , and by NASA on the Space Shuttle during the 1980s . The GRiD 's manufacturer subsequently earned significant returns on its patent rights as its
innovations became commonplace . GRiD Systems Corp. was later bought by the Tandy ( now RadioShack ) Corporation . Dulmont Magnum / Kookaburra ( edit ) Main
article : Dulmont Magnum Another contender for the first true laptop was the Dulmont Magnum , designed Barry Wilkinson and Terry Crews Engineering Manager at
Dulmison in 1982 and released in Australia in 1983 It included an 80 × 8 display in a lid that closed against the keyboard . It was based on the MS - DOS
operating system and applications stored in ROM ( A :) and also supported removable modules in expansion slots ( B : and C :) that could be custom programmed
EPROM or standard word processing and spreadsheet applications .
</item 6>

<item 7>
History of laptops
They also introduced the now - standard `` resume '' feature to DOS - based machines : the computer could be paused between sessions without having to be
restarted each time . U.S. Air Force ( edit ) The first laptops successful on a large scale came in large part due to a Request For Proposal ( RFP ) by the U.S.
Air Force in 1987 . This contract would eventually lead to the purchase of over 200,000 laptops . Competition to supply this contract was fiercely contested and
the major PC companies of the time ; IBM Corporation , To shins , Compaq , NEC , and Zenith Data Systems ( ZDS ) , rushed to develop laptops in an attempt to
win this deal . ZDS , which had earlier won a landmark deal with the IRS for its Z - 171 , was awarded this contract for its SupersPort series . The SupersPort
series was originally launched with an Intel 8086 processor , dual floppy disk drives , a backlit , blue and white STN LCD screen , and a NiCd battery pack .
Later models featured an Intel 80286 processor and a 20 MB hard disk drive .
</item 7>

<item 8>
History of laptops
IBM PC Convertible ( edit ) Also among the first commercial IBM - compatible laptops was the IBM PC Convertible , introduced in 1986 . It had a CGA - compatible
LCD and 2 floppy drives . It weighed 13 pounds ( 5.9 kg ) . Toshiba T1100 , T1000 , and T1200 ( edit ) Toshiba launched the Toshiba T1100 in 1985 , and has
subsequently described it as `` the world 's first mass - market laptop computer '' . It did not have a hard drive , and ran entirely from floppy disks . The
CPU was a 4.77 MHz Intel 80C88 , a variation of the popular Intel 8088 , and the display was a monochrome , text - only 640x200 LCD . It was followed in 1987 by
the T1000 and T1200 . Although limited floppy - based DOS machines , with the operating system stored in ROM , the Toshiba models were small and light enough to
be carried in a backpack , and could be run from lead - acid batteries .
</item 8>

<item 9>
History of laptops
Before laptop / notebook computers were technically feasible , similar ideas had been proposed , most notably Alan Kay 's Dynabook concept , developed at Xerox
PARC in the early 1970 s . One of the first reasonably portable computers was the Xerox NoteTaker , again developed at Xerox PARC , in 1976 . However , only 10
prototypes were built . Contents ( hide ) 1 Portal R2E CCMC 2 Osborne 1 3 Bondwell 2 4 Compaq Portable 5 Epson HX - 20 6 Grid Compass 7 Dulmont Magnum /
Kookaburra 8 Ampere 9 Tandy Model 100 10 Sharp and Gavilan 11 Kyotronic 85 12 Commodore SX - 64 13 Kaypro 2000 14 IBM PC Convertible 15 Toshiba T1100 , T1000 ,
and T1200 16 US Air Force 17 Hewlett - Packard Vectra Portable CS 18 Cambridge Z88 19 Compaq SLT / 286 20 NEC UltraLite 21 Apple 21.1 Macintosh Portable 21.
</item 9>

<item 10>
Graph query row:
entity: Toshiba T1100
relationship: CREATED
related: Toshiba
</item 10>
=== END ITEM id=d2b24b09b46e9386 ===

=== ITEM id=45d4acb19ab5b8bf ===
Gold sentences and facts:
Sentence 0: Natural Born Killers Theatrical release poster Directed by Oliver Stone Produced by Jane Hamsher Don Murphy Clayton Townsend Screenplay by Richard Rutowski Oliver Stone David Veloz Story by Quentin Tarantino Starring Woody Harrelson Juliette Lewis Robert Downey , Jr . Tommy Lee Jones Tom Sizemore Cinematography Robert Richardson Edited by Hank Corwin Brian Berdan Production company Regency Enterprises Distributed by Warner Bros . Release date August 26 , 1994 ( 1994 - 08 - 26 ) Running time 119 minutes Country United States Language English Japanese Navajo Budget $34 million Box office $50.3 million
  Fact 0.0: Natural Born Killers has a theatrical release poster.
  Fact 0.1: Natural Born Killers was directed by Oliver Stone.
  Fact 0.2: Natural Born Killers was produced by Jane Hamsher.
  Fact 0.3: Natural Born Killers was produced by Don Murphy.
  Fact 0.4: Natural Born Killers was produced by Clayton Townsend.
  Fact 0.5: Natural Born Killers has a screenplay by Richard Rutowski.
  Fact 0.6: Natural Born Killers has a screenplay by Oliver Stone.
  Fact 0.7: Natural Born Killers has a screenplay by David Veloz.
  Fact 0.8: Natural Born Killers has a story by Quentin Tarantino.
  Fact 0.9: Natural Born Killers stars Woody Harrelson.
  Fact 0.10: Natural Born Killers stars Juliette Lewis.
  Fact 0.11: Natural Born Killers stars Robert Downey, Jr.
  Fact 0.12: Natural Born Killers stars Tommy Lee Jones.
  Fact 0.13: Natural Born Killers stars Tom Sizemore.
  Fact 0.14: Natural Born Killers has cinematography by Robert Richardson.
  Fact 0.15: Natural Born Killers was edited by Hank Corwin.
  Fact 0.16: Natural Born Killers was edited by Brian Berdan.
  Fact 0.17: Natural Born Killers was produced by Regency Enterprises.
  Fact 0.18: Natural Born Killers was distributed by Warner Bros.
  Fact 0.19: Natural Born Killers has a release date of August 26, 1994.
  Fact 0.20: Natural Born Killers has a running time of 119 minutes.
  Fact 0.21: Natural Born Killers is from the United States.
  Fact 0.22: Natural Born Killers has English, Japanese and Navajo languages.
  Fact 0.23: Natural Born Killers had a budget of $34 million.
  Fact 0.24: Natural Born Killers had a box office of $50.3 million.
Sentence 1: Natural Born Killers is a 1994 American satirical black comedy crime film directed by Oliver Stone and starring Woody Harrelson , Juliette Lewis , Robert Downey Jr. , Tom Sizemore , and Tommy Lee Jones . The film was released in the United States on August 26 , 1994 . The film tells the story of two victims of traumatic childhoods who became lovers and mass murderers , and are irresponsibly glorified by the mass media .
  Fact 1.0: Natural Born Killers is a 1994 American satirical black comedy crime film.
  Fact 1.1: Natural Born Killers was directed by Oliver Stone.
  Fact 1.2: Natural Born Killers stars Woody Harrelson.
  Fact 1.3: Natural Born Killers stars Juliette Lewis.
  Fact 1.4: Natural Born Killers stars Robert Downey Jr.
  Fact 1.5: Natural Born Killers stars Tom Sizemore.
  Fact 1.6: Natural Born Killers stars Tommy Lee Jones.
  Fact 1.7: Natural Born Killers was released in the United States on August 26, 1994.
  Fact 1.8: The film tells the story of two victims of traumatic childhoods.
  Fact 1.9: The two victims became lovers.
  Fact 1.10: The two victims became mass murderers.
  Fact 1.11: The two victims are irresponsibly glorified by the mass media.

Retrieved items:
<item 1>
Natural Born Killers
Natural Born Killers - Wikipedia Natural Born Killers Jump to : navigation , search Not to be confused with Natural Born Killaz . Natural Born Killers
Theatrical release poster Directed by Oliver Stone Produced by Jane Hamsher Don Murphy Clayton Townsend Screenplay by Richard Rutowski Oliver Stone David Veloz
Story by Quentin Tarantino Starring Woody Harrelson Juliette Lewis Robert Downey , Jr . Tommy Lee Jones Tom Sizemore Cinematography Robert Richardson Edited by
Hank Corwin Brian Berdan Production company Regency Enterprises Distributed by Warner Bros . Release date August 26 , 1994 ( 1994 - 08 - 26 ) Running time 119
minutes Country United States Language English Japanese Navajo Budget $34 million Box office $50.3 million Natural Born Killers is a 1994 American satirical
black comedy crime film directed by Oliver Stone and starring Woody Harrelson , Juliette Lewis , Robert Downey Jr. , Tom Sizemore , and Tommy Lee Jones . The
film was released in the United States on August 26 , 1994 .
</item 1>

<item 2>
Natural Born Killers
Emil Reingold Pruitt Taylor Vince as Deputy Warden Kavanaugh Joe Grifasi as Deputy Sheriff Duncan Homolka Everett Quinton as Deputy Wurlitzer Marshall Bell as
Deputy Peter Crombie as Intense cop Grand L. Bush as Prison inmate Louis Lombardi as Deputy `` Sparky '' Dale Dye as Dale Wrigley Corey Everson as Mallory Knox
in TV reconstruction O - Lan Jones as Mabel James Gammon ( uncredited ) as Redneck buddy Mark Harmon ( uncredited ) as Mickey Knox in TV reconstruction Adrien
Brody and David Pasquesi ( uncredited ) as Gale 's cameramen Arliss Howard ( uncredited ) as Owen Traft , Mickey and Mallory 's Guardian angel / The Demon
Deleted scenes Ashley Judd as Grace Mulberry Rachel Ticotin as Prosecutor Wanda Bisbing Peter and David Paul as Simon and Norman Hun Denis Leary and Bret Hart
as Prison inmates Production Natural Born Killers was based upon a screenplay written by Quentin Tarantino , in which a married couple suddenly decide to go on
a killing spree .
</item 2>

<item 3>
Natural Born Killers
Entertainment Weekly ranked the film as the eighth most controversial film ever . `` Copycat '' crimes Main article : Natural Born Killers copycat crimes From
almost the moment of its release , the film has been accused of encouraging and inspiring numerous murderers in North America , including the Heath High School
shooting and the Columbine High School massacre . Soundtrack Main article : Natural Born Killers ( soundtrack ) The soundtrack was released August 23 , 1994 by
Interscope Records . `` Waiting for the Miracle '' ( Edit ) -- Leonard Cohen `` Shitlist '' -- L7 `` Moon Over Greene County '' ( Edit ) -- Dan Zanes `` Rock N
Roll Nigger '' ( Flood Remix ) -- Patti Smith `` Sweet Jane '' ( Edit ) -- Cowboy Junkies `` You Belong to Me '' -- Bob Dylan `` The Trembler '' ( Edit ) --
Duane Eddy `` Burn '' -- Nine Inch Nails `` Route 666 '' -- BB Tone Brian Berdan featuring Robert Downey Jr .
</item 3>

<item 4>
Natural Born Killers
Style Natural Born Killers is shot and edited in a frenzied and psychedelic style consisting of black and white , animation , and other unusual color schemes ,
and employing a wide range of camera angles , filters , film stocks , lenses , and special effects . Much of the film is told via parodies of television shows ,
including a scene ( I Love Mallory ) presented in the style of a sitcom about a dysfunctional family . Commercials which were commonly on the air at the time of
the film 's release make brief , intermittent appearances . In his DVD director 's commentary , Stone goes into great detail about the look of the film ,
explaining scene by scene why a particular look was chosen for a particular scene . According to Hollywood.com , Natural Born Killers is a satirical crime film
, while Foster Hirsch deemed it `` the crime film as would - be social and cultural satire '' . Stone considered it his road film , specifically naming Bonnie
and Clyde as a source of inspiration .
</item 4>

<item 5>
Natural Born Killers
Nytimes.com ( 1996 - 04 - 14 ) . Retrieved on 2014 - 05 - 22 . Jump up ^ Natural Born Killers ( Unrated Director 's Cut ) ( Blu - ray ) : Woody Harrelson ,
Juliette Lewis , Tom Sizemore , Oliver Stone : Movies & TV . Amazon.com . Retrieved on 2014 - 05 - 22 . Jump up ^ `` ' Natural Born Killers ' Is Banned in
Ireland '' . NY Times . Retrieved 2013 - 05 - 26 . Jump up ^ `` Irish Film Institute - Natural Born Killers '' . Irish Film Institute . Retrieved 2013 - 05 - 26
. Jump up ^ ( 1 ) Archived March 19 , 2015 , at the Wayback Machine . Jump up ^ `` Natural Born Killers given video release '' . The Guardian . London . May 16
, 2001 .
</item 5>

<item 6>
Natural Born Killers
Dre `` Born Bad '' -- Juliette Lewis `` Fall of the Rebel Angels '' ( Edit ) -- Sergio Cervetti `` Forkboy '' -- Lard `` Batonga in Batongaville '' ( Edit ) --
`` A Warm Place '' ( Edit ) -- Nine Inch Nails `` Allah , Mohammad , Char , Yaar '' -- Nusrat Fateh Ali Khan `` The Future '' ( Edit ) -- Leonard Cohen `` What
Would U Do ? '' -- Tha Dogg Pound Tracks 10 , 13 , 18 , 20 , 23 , and 25 are assembled from various recordings and dialogue from the film . See also Film in the
United States portal 1990s portal Charles Starkweather References Jump up ^ `` NATURAL BORN KILLERS ( 18 ) '' . British Board of Film Classification . December
20 , 1994 . Retrieved September 25 , 2014 . ^ Jump up to : `` Natural Born Killers ( 1994 ) '' . Box Office Mojo. 1994 - 10 - 18 . Retrieved 2016 - 01 - 06 .
</item 6>

<item 7>
Natural Born Killers
McClusky and his guards are then quickly massacred by hordes of inmates . Mickey and Mallory steal a van and kill the last guard ; Owen 's fate is unknown .
Escaping to a rural location , they give a final interview to Gale , whose ear they have seemingly removed ( implied , but as a camera track reveals ) , before
they tell him he must die also . He attempts various arguments to change their minds , finally appealing to their trademark practice of leaving one survivor ;
Mickey informs him they are leaving a witness to tell the tale , his camera . Gale accepts his fate and extends his arms as if on a cross as they shoot him dead
while his unattended camera continues to roll . The couple is shown several years later , in an RV , with Mickey driving and a pregnant Mallory watching their
two children play . Cast Main article : List of Natural Born Killers characters Woody Harrelson as Mickey Knox Juliette Lewis as Mallory Wilson Knox Robert
Downey , Jr. as Wayne Gale Tom Sizemore as Detective Jack Scagnetti Tommy Lee Jones as Warden Dwight McClusky Rodney Dangerfield as Ed Wilson Edie McClurg as
Mrs.
</item 7>

<item 8>
Natural Born Killers
( 2017 ) Retrieved from `` https://en.wikipedia.org/w/index.php?title=Natural_Born_Killers&oldid=819366442 '' Categories : 1994 films 1990s crime films 1990s
drama films American films American black comedy films American crime drama films American independent films American satirical films English - language films
Film duos Films about psychopaths Films directed by Oliver Stone Films set in the 1990s Films set in Arizona Films set in Illinois Films set in Indiana Films
set in New Mexico Films shot in multiple formats Films with live action and animation Incest in film Matricide in fiction Obscenity controversies in film
Patricide in fiction Regency Enterprises films Road movies Screenplays by Oliver Stone Screenplays by Quentin Tarantino Serial killer films Venice Grand Special
Jury Prize winners Warner Bros.
</item 8>

<item 9>
Natural Born Killers
The film was released in the United States on August 26 , 1994 . The film tells the story of two victims of traumatic childhoods who became lovers and mass
murderers , and are irresponsibly glorified by the mass media . The film is based on an original screenplay by Quentin Tarantino that was heavily revised by
writer David Veloz , associate producer Richard Rutowski , and director Stone ; Tarantino received story credit . Jane Hamsher , Don Murphy , and Clayton
Townsend produced the film , with Arnon Milchan , Thom Mount , and Stone as executive producers . The film was released theatrically on August 26 , 1994 in the
United States , February 24 , 1995 and premiered on Venice Film Festival on August 29 , 1994 . The film was a box office success , grossing over $50 million
against a production budget of $34 million , and received generally mixed to positive reviews from critics , who praised the cast 's performances , the movie 's
humor , plot , and combination of action and romance , but found the film much too violent and graphic .
</item 9>

<item 10>
Natural Born Killers
Filming locations included the Rio Grande Gorge Bridge just west of Taos , New Mexico , where the wedding scene was filmed , and Stateville Correctional Center
in Joliet , Illinois , where the prison riot was filmed . In Stateville , 80 % of the prisoners are incarcerated for violent crimes . For the first two weeks on
location at the prison , the extras were actual inmates with rubber weapons . For the subsequent two weeks , 200 extras were needed because the Stateville
inmates were on lockdown . According to Tom Sizemore , during filming on the prison set , Stone would play African tribal music at full blast between takes to
keep the frantic energy up . Whilst shooting the POV scene wherein Mallory runs into the wire mesh , director of photography Robert Richardson broke his finger
and the replacement cameraman cut his eye . According to Oliver Stone , he was n't too popular with the camera department on set that day . For the scenes
involving rear projection , the projected footage was shot prior to principal photography , then edited together , and projected onto the stage , behind the
live actors .
</item 10>
=== END ITEM id=45d4acb19ab5b8bf ===

=== ITEM id=c11303ee76c76355 ===
Gold sentences and facts:
Sentence 0: `` Where Have All the Flowers Gone ? '' is a modern folk - style song . The melody and the first three verses were written by Pete Seeger in 1955 and published in Sing Out ! magazine . Additional verses were added in May 1960 by Joe Hickerson , who turned it into a circular song . Its rhetorical `` where ? '' and meditation on death place the song in the ubi sunt tradition . In 2010 , the New Statesman listed it as one of the `` Top 20 Political Songs '' .
  Fact 0.0: Where Have All the Flowers Gone? is a modern folk-style song.
  Fact 0.1: The melody of Where Have All the Flowers Gone? was written by Pete Seeger in 1955.
  Fact 0.2: The first three verses of Where Have All the Flowers Gone? were written by Pete Seeger in 1955.
  Fact 0.3: The melody of Where Have All the Flowers Gone? was published in Sing Out! magazine.
  Fact 0.4: The first three verses of Where Have All the Flowers Gone? were published in Sing Out! magazine.
  Fact 0.5: Additional verses of Where Have All the Flowers Gone? were added in May 1960 by Joe Hickerson.
  Fact 0.6: Joe Hickerson turned Where Have All the Flowers Gone? into a circular song.
  Fact 0.7: The rhetorical 'where?' and meditation on death of Where Have All the Flowers Gone? place the song in the ubi sunt tradition.
  Fact 0.8: In 2010, the New Statesman listed Where Have All the Flowers Gone? as one of the 'Top 20 Political Songs'.

Retrieved items:
<item 1>
“Where Have All the Flowers Gone?” is a modern folk song whose melody and first three verses Pete Seeger wrote in 1955 and published in *Sing Out!* Joe
Hickerson added verses in May 1960, making it a circular song. Its repeated questions and meditation on death connect it to the *ubi sunt* tradition. A 1960
recording appeared on Seeger’s *The Rainbow Quest*; the song was named one of the New Statesman’s top 20 political songs
</item 1>

<item 2>
Where Have All the Flowers Gone?
The 1964 release of the song as a Columbia Records 45 single , 13 - 33088 , by Pete Seeger was inducted into the Grammy Hall of Fame in 2002 in the Folk
category . Contents ( hide ) 1 Composition 2 Versions 3 Grammy Hall of Fame 4 See also 5 References 6 Bibliography Composition ( edit ) Seeger found inspiration
for the song in October 1955 while he was on a plane bound for a concert at Oberlin College , one of the few venues which would hire him during the McCarthy era
. Leafing through his notebook he saw the passage , `` Where are the flowers , the girls have plucked them . Where are the girls , they 've all taken husbands .
Where are the men , they 're all in the army . '' These lines were taken from the traditional Cossack folk song `` Koloda - Duda '' , referenced in the Mikhail
Sholokhov novel And Quiet Flows the Don ( 1934 ) , which Seeger had read `` at least a year or two before '' . Seeger created a song which was subsequently
published in Sing Out in 1962 .
</item 2>

<item 3>
“Where Have All the Flowers Gone?” is an anti-war folk song written by Pete Seeger and first recorded in 1955. The material lists recordings and related works
by Seeger and The Weavers, plus versions by Marlene Dietrich, Johnny Rivers, The Searchers, and The Kingston Trio. It also includes Seeger’s family and
activism, Dietrich’s albums and singles, and links to related discographies and articles.
</item 3>

<item 4>
“Where Have All the Flowers Gone?” inspired numerous recordings: Johnny Rivers’ 1965 folk-rock hit reached No. 26 in the US and No. 9 in Canada. Grady Martin,
Harry Belafonte, Lester Flatt and Earl Scruggs, Walter Jackson, Wes Montgomery, the Landsmen, and Bill Anderson also recorded it. Rufus Harley made a
bagpipe-jazz version in 1969, released posthumously in 2006.
</item 4>

<item 5>
Where Have All the Flowers Gone?
Where Have All the Flowers Gone ? - Wikipedia Where Have All the Flowers Gone ? Jump to : navigation , search For the 2008 documentary , see Where Have All the
Flowers Gone ? ( film ) . 1960 release on The Rainbow Quest LP , Folkways , FA 2454 , by Pete Seeger Columbia Hall of Fame 45 rpm single release as 13 - 33088
featuring the November 1962 version . August 1965 . `` Where Have All The Flowers Gone ? '' Song Language English Composer ( s ) Pete Seeger Lyricist ( s ) Pete
Seeger and Joe Hickerson `` Where Have All the Flowers Gone ? '' is a modern folk - style song . The melody and the first three verses were written by Pete
Seeger in 1955 and published in Sing Out ! magazine . Additional verses were added in May 1960 by Joe Hickerson , who turned it into a circular song . Its
rhetorical `` where ? '' and meditation on death place the song in the ubi sunt tradition . In 2010 , the New Statesman listed it as one of the `` Top 20
Political Songs '' .
</item 5>

<item 6>
Where Have All the Flowers Gone?
Versions ( edit ) The Kingston Trio recorded the song in 1961 . Believing it to be a traditional song , they claimed authorship , although upon notice from
Seeger they had their name removed and credited Seeger and Hickerson . Seeger acknowledged their success with this song . Their single , with `` O Ken Karanga
'' as the A-side and the hit `` Where Have All The Flowers Gone ? '' the B - side , reached # 21 in the 1962 Billboard Hot 100 chart and # 4 on the Easy
Listening chart . Peter , Paul and Mary included the song on their eponymous debut album ( which spent five weeks as the # 1 album in the country ) in 1962 .
Marlene Dietrich performed the song in English , French , and German . The song was first performed in French ( as `` Qui peut dire où vont les fleurs ? '' ) by
Marlene in 1962 at a UNICEF concert . She also recorded the song in English and in German , the latter titled `` Sag ' mir , wo die Blumen sind '' , with lyrics
translated by Max Colpet .
</item 6>

<item 7>
“Where Have All the Flowers Gone?” has been recorded in many styles and languages. Versions include Chris de Burgh’s 2008 recording, sung at Harry Patch’s 2009
funeral; instrumental recordings by the Folkswingers and Sharon Isbin; and versions by Jimmy Somerville, The Tansads, Einstürzende Neubauten, and Lara Veronin.
Polish, Czech, Hungarian, and Croatian adaptations were also made, while Kirsten Hasberg created a parody
</item 7>

<item 8>
Where Have All the Flowers Gone?
Sing Out ! . 53 ( 2 ) : 76 . ^ Jump up to : Joe Hickerson . `` Where Have All the Flowers Gone ? '' . presentation to SEM ( Society for Ethnomusicology ) , 50th
Annual Meeting in Atlanta ( quoted in thread ) . Mudcat.org . Retrieved October 20 , 2009 . ^ Jump up to : Smith , Ian K ( March 25 , 2010 ) . `` Top 20
Political Songs : Where Have All the Flowers Gone '' . New Statesman . Retrieved March 25 , 2010 . Jump up ^ A Folk Legend 's Fertile Ground , Oberlin Alumni
Magazine , summer 2014 Jump up ^ Notes from : Where Have All the Flowers Gone - The Songs of Pete Seeger ^ Jump up to : Dunaway , David King ( 2008 ) . How Can
I Keep From Singing ? The Ballad of Pete Seeger , pp. 228 - 30 . Random House , Inc . ISBN 0 - 345 - 50608 - 1 .
</item 8>

<item 9>
Where Have All the Flowers Gone?
Version Title Artist Basque Loreak non dira ? Lou Topet , Harkaitz Cano Belarusian Дзе кветкі ўсе ? Bar Akaryna Bengali কোথায় গেল তারা ? ( Kothay Gelo Tara )
Kabir Suman Catalan Què se n'ha fet d'aquelles flors ? Roslyn Smith , La Marta ( Club Super3 ) Chinese 花 兒 怎麼 不見 了 ? Poon Sow Keng ( 潘秀瓊 ) Croatian Kamo je
cvijeće otišlo ? Monia Verardi Czech Řekni , kde ty kytky jsou Judita Čeřovská , Marie Rottrová , Marta Kubišová Danish Where Have All the Flowers Gone Savage
Rose Dutch Zeg me waar de bloemen zijn Jaap Fischer English Where have all the flowers gone ?
</item 9>

<item 10>
Pete Seeger wrote “Where Have All the Flowers Gone?” after finding a verse in his notebook while traveling to a 1955 concert at Oberlin College. The verse came
from the traditional Cossack song “Koloda-Duda,” encountered in Mikhail Sholokhov’s *And Quiet Flows the Don*. Seeger developed the song and published it in
*Sing Out!* in 1962. His 1964 Columbia single was inducted into the Grammy Hall of Fame
</item 10>
=== END ITEM id=c11303ee76c76355 ===

=== ITEM id=1e092fd24eabb09e ===
Gold sentences and facts:
Sentence 0: India and Pakistan acquired independence in 1947 , but Governors - General continued to be appointed over each nation until republican constitutions were written . Louis Mountbatten , 1st Earl Mountbatten of Burma remained Governor - General of India for some time after independence , but the two nations were otherwise headed by native Governors - General . India became a secular republic in 1950 ; Pakistan became an Islamic one in 1956 .
  Fact 0.0: India and Pakistan gained independence in 1947.
  Fact 0.1: Governors-General continued to be appointed over each nation after 1947 until republican constitutions were written.
  Fact 0.2: Louis Mountbatten, 1st Earl Mountbatten of Burma, remained Governor-General of India for some time after independence.
  Fact 0.3: The two nations were otherwise headed by native Governors-General.
  Fact 0.4: India became a secular republic in 1950.
  Fact 0.5: Pakistan became an Islamic republic in 1956.

Retrieved items:
<item 1>
Governor-General of India
The first Governor - General of British India was Warren Hastings , and the first Governor - General of independent India was Louis Mountbatten . Contents (
hide ) 1 History 2 Functions 3 Council 4 Style and title 5 Flag 6 Residence 7 Insignia 8 See also 9 References 10 Further reading History ( edit ) Warren
Hastings , the first Governor - General of Fort William from 1773 to 1785 . Many parts of the Indian subcontinent were governed by the East India Company ,
which nominally acted as the agent of the Mughal Emperor . In 1773 , motivated by corruption in the Company , the British government assumed partial control
over the governance of India with the passage of the Regulating Act of 1773 . A Governor - General and Supreme Council of Bengal were appointed to rule over the
Presidency of Fort William in Bengal . The first Governor - General and Council were named in the Act . The Charter Act 1833 replaced the Governor - General and
Council of Fort William with the Governor - General and Council of India .
</item 1>

<item 2>
Governor-General of India
The power to elect the Governor - General was retained by the Court of Directors , but the choice became subject to the Sovereign 's approval . After the Indian
Rebellion of 1857 , the East India Company 's territories in India were put under the direct control of the Sovereign . The Government of India Act 1858 vested
the power to appoint the Governor - General in the Sovereign . The Governor - General , in turn , had the power to appoint all lieutenant governors in India ,
subject to the Sovereign 's approval . India and Pakistan acquired independence in 1947 , but Governors - General continued to be appointed over each nation
until republican constitutions were written . Louis Mountbatten , 1st Earl Mountbatten of Burma remained Governor - General of India for some time after
independence , but the two nations were otherwise headed by native Governors - General . India became a secular republic in 1950 ; Pakistan became an Islamic
one in 1956 . Functions ( edit ) Lord Curzon in his robes as Viceroy of India , a post he held from 1899 to 1905 .
</item 2>

<item 3>
Governor-General of India
Throughout the British administration , Governors - General retreated to the Viceregal Lodge ( Rashtrapati Niwas ) at Shimla each summer to escape the heat ,
and the government of India moved with them . The Viceregal Lodge now houses the Indian Institute of Advanced Study . Insignia ( edit ) See also : Star of India
( flag ) Badge of the Governor - General ( 1885 -- 1947 ) Standard of the Governor - General ( 1885 -- 1947 ) Standard of the Governor - General ( 1947 -- 50 )
See also ( edit ) India portal Pakistan portal United Kingdom portal Politics portal List of governors - general of India Commander - in - Chief , India British
Empire Emperor of India Indian independence movement Council of India British Raj Secretary of State for India India Office Indian Civil Service Partition of
India History of Bangladesh History of India History of Pakistan References ( edit ) Jump up ^ The term British India is mistakenly used to mean the same as the
British Indian Empire , which included both the provinces and the Native States . Jump up ^ `` Imperial Impressions '' . Hindustan Times . 20 July 2011 .
</item 3>

<item 4>
Governor-General of India
The Chamber of Princes was an institution established in 1920 by a Royal Proclamation of King - Emperor George V to provide a forum in which the princely rulers
could voice their needs and aspirations to the government . The chamber usually met only once a year , with the Viceroy presiding , but it appointed a Standing
Committee , which met more often . Upon independence in August 1947 , the title of Viceroy was abolished . The representative of the British Sovereign became
known once again as the Governor - General . C. Rajagopalachari became the only Indian Governor - General . However , once India acquired independence , the
Governor - General 's role became almost entirely ceremonial , with power being exercised on a day - to - day basis by the Indian cabinet . After the nation
became a republic in 1950 , the President of India continued to perform the same functions .
</item 4>

<item 5>
Governor-General of India
Lord Mountbatten addressing the Chamber of Princes as Crown Representative in the 1940s The Governor - General originally had power only over the Presidency of
Fort William in Bengal . The Regulating Act , however , granted them additional powers relating to foreign affairs and defence . The other Presidencies of the
East India Company ( Madras , Bombay and Bencoolen ) were not allowed to declare war on or make peace with an Indian prince without receiving the prior approval
of the Governor - General and Council of Fort William . The powers of the Governor - General , in respect of foreign affairs , were increased by the India Act
1784 . The Act provided that the other Governors under the East India Company could not declare war , make peace or conclude a treaty with an Indian prince
unless expressly directed to do so by the Governor - General or by the Company 's Court of Directors . While the Governor - General thus became the controller
of foreign policy in India , he was not the explicit head of British India . That status came only with the Charter Act 1833 , which granted him ``
superintendence , direction and control of the whole civil and military Government '' of all of British India .
</item 5>

<item 6>
Governor-General of India
The title of viceroy was abandoned when British India split into the two independent dominions of India and Pakistan , but the office of governor - general
continued to exist in each country separately -- until they adopted republican constitutions in 1950 and 1956 , respectively . Until 1858 , the governor -
general was selected by the Court of Directors of the East India Company , to whom he was responsible . Thereafter , he was appointed by the sovereign on the
advice of the British government ; the Secretary of State for India , a member of the UK Cabinet , was responsible for instructing him on the exercise of his
powers . After 1947 , the sovereign continued to appoint the governor - general , but did so on the advice of the Indian government . Governors - General served
at the pleasure of the sovereign , though the practice was to have them serve five - year terms . Governors - General could have their commission rescinded ;
and if one was removed , or left , a provisional governor - general was sometimes appointed until a new holder of the office could be chosen .
</item 6>

<item 7>
Governor-General of India
Most Governors - General and Viceroys were peers . Frequently , a Viceroy who was already a peer would be granted a peerage of higher rank , as with the
granting of a marquessate to Lord Reading and an earldom and later a marquessate to Freeman Freeman - Thomas . Of those Viceroys who were not peers , Sir John
Shore was a baronet , and Lord William Bentinck was entitled to the courtesy title ' Lord ' because he was the son of a Duke . Only the first and last Governors
- General -- Warren Hastings and Chakravarti Rajagopalachari -- as well as some provisional Governors - General , had no honorific titles at all . Flag ( edit )
Flag of the Governor General of India ( 1947 -- 50 ) From around 1885 , the Viceroy of India was allowed to fly a Union Flag augmented in the centre with the '
Star of India ' surmounted by a Crown . This flag was not the Viceroy 's personal flag ; it was also used by Governors , Lieutenant Governors , Chief
Commissioners and other British officers in India .
</item 7>

<item 8>
Governor-General of India
The officer had direct control only over Fort William , but supervised other British East India Company officials in India . Complete authority over all of
British India was granted in 1833 , and the official came to be known as the `` Governor - General of India '' . In 1858 , the territories of the East India
Company came under the direct control of the British government ; see British Raj . The governor - general ( now also the viceroy ) headed the central
government of India , which administered the provinces of British India , including the Punjab , Bengal , Bombay , Madras , the United Provinces , and others .
However , much of India was not ruled directly by the British government ; outside the provinces of British India , there were hundreds of nominally sovereign
princely states or `` native states '' , whose relationship was not with the British government , but directly with the monarch . To reflect the governor -
general 's role as the representative of the monarch to the feudal rulers of the princely states , from 1858 the term Viceroy and Governor - General of India (
known in short as the Viceroy of India ) was applied to him .
</item 8>

<item 9>
The Governor-General’s office began under the East India Company, with Warren Hastings its first holder in 1774. Its authority expanded from Bengal to all
British India in 1833. After Company rule ended in 1858, the officeholder also represented the monarch and was commonly called the Viceroy. Following
independence in 1947, India and Pakistan retained separate, largely ceremonial governors-general until becoming republics. Louis Mountbatten was independent
India’s first; C. Rajagopalach
</item 9>

<item 10>
Governor-General of India
Governor - General of India - Wikipedia Governor - General of India Jump to : navigation , search This article is about the position of Governor - General of
India . For individual Governor - Generals , see List of governors - general of India . Viceroy and Governor - General of India Standard of the Governor -
General Louis Mountbatten , the last Viceroy of India & the first Governor - General during the dominion period Style His Excellency Residence Viceroy 's House
Appointer East India Company ( to 1858 ) Monarch of India ( from 1858 ) Formation 20 October 1774 First holder Warren Hastings Final holder Chakravarthi
Rajagopalachari Abolished 26 January 1950 The Governor - General of India ( or , from 1858 to 1947 , officially the Viceroy and Governor - General of India ,
commonly shortened to Viceroy of India ) was originally the head of the British administration in India and , later , after Indian independence in 1947 , the
representative of the Indian head of state . The office was created in 1773 , with the title of Governor - General of the Presidency of Fort William .
</item 10>
=== END ITEM id=1e092fd24eabb09e ===

=== ITEM id=cbaf3b8d27603b0a ===
Gold sentences and facts:
Sentence 0: Year Winning team Losing team Score Site Notes 1909 Pittsburgh Pirates Detroit Tigers 8 -- 0 Bennett Park , Detroit Rookie Babe Adams tosses a six hit shutout for his third win of the series . 1912 Boston Red Sox New York Giants 3 -- 2 F / 10 Fenway Park , Boston This decisive contest was actually game eight , as game two had ended in a tie . 1924 Washington Senators New York Giants 4 -- 3 F / 12 Griffith Stadium , Washington , D.C. Longest game seven in innings . In the 12th , Muddy Ruel doubled after a dropped foul popfly by Giants catcher Hank Gowdy . Ruel scored on a game - winning RBI double by Earl McNeely . 1925 Pittsburgh Pirates Washington Senators 9 -- 7 Forbes Field , Pittsburgh Pittsburgh became the first team in a best of seven to win the series when trailing three games to one . 1926 St. Louis Cardinals New York Yankees 3 -- 2 Yankee Stadium , the Bronx , New York City Grover Cleveland Alexander strikes out Tony Lazzeri with the bases loaded to end the seventh and preserve the Cardinals lead . Two innings later Babe Ruth is caught stealing to end the World Series . 1931 St. Louis Cardinals Philadelphia Athletics 4 -- 2 St. Louis , Missouri Last postseason game for the Athletics while based in Philadelphia . 1934 St. Louis Cardinals Detroit Tigers 11 -- 0 Navin Field , Detroit Dizzy Dean 's second win , a six - hit shutout on one day of rest , gives the Cardinals the title . 1940 Cincinnati Reds Detroit Tigers 2 -- 1 Crosley Field , Cincinnati This game was over in a snappy 1 : 47 while a sacrifice bunt and sacrifice fly scored the winning runs . 1945 Detroit Tigers Chicago Cubs 9 -- 3 Wrigley Field , Chicago Cubs last World Series appearance until 2016 . See also : Curse of the Billy Goat 1946 St. Louis Cardinals Boston Red Sox 4 -- 3 Sportsman 's Park III , St. Louis , Missouri Enos Slaughter 's famous mad dash scored the deciding run to win the game and the series . 1947 New York Yankees Brooklyn Dodgers 5 -- 2 Yankee Stadium , the Bronx , New York City This was the Series that featured Cookie Lavagetto 's pinch - hit that spoiled Bill Bevens ' no - hit bid and Al Gionfriddo 's catch to rob Joe DiMaggio of a home run . 1952 New York Yankees Brooklyn Dodgers 4 -- 2 Ebbets Field , Brooklyn , New York City Billy Martin 's game saving infield catch preserves the Yankees lead and helps win the game and the series . 1955 Brooklyn Dodgers New York Yankees 2 -- 0 Yankee Stadium , the Bronx , New York City Dodgers ' only championship in Brooklyn . The home team won all games of the series , except for game seven . 1956 New York Yankees Brooklyn Dodgers 9 -- 0 Ebbets Field , Brooklyn , New York City Jackie Robinson 's final Major League game . Last postseason game at Ebbets Field and Dodgers ' last postseason game before moving to Los Angeles . The home team won all games of the series , except for game seven . 1957 Milwaukee Braves New York Yankees 5 -- 0 Yankee Stadium , the Bronx , New York City The Braves ' only championship in Milwaukee . 1958 New York Yankees Milwaukee Braves 6 -- 2 County Stadium , Milwaukee The Yankees win their 18th World Series title after being down three games to one . This was the Braves ' last postseason game in Milwaukee . 1960 Pittsburgh Pirates New York Yankees 10 -- 9 Forbes Field , Pittsburgh Bill Mazeroski hit the walk - off home run to win the Pirates the championship . Last postseason game in Forbes Field . Last postseason game for Yankees ' manager Casey Stengel . Highest scoring game seven . 1962 New York Yankees San Francisco Giants 1 -- 0 Candlestick Park , San Francisco With runners on second and third and two outs , Willie McCovey 's line drive to Bobby Richardson ended the game . First game seven that ended with a 1 - 0 score . 1964 St. Louis Cardinals New York Yankees 7 -- 5 Busch Stadium I , St. Louis , Missouri The Yankees ' last World Series appearance until 1976 . Last postseason game at Busch Stadium I . 1965 Los Angeles Dodgers Minnesota Twins 2 -- 0 Metropolitan Stadium , Bloomington , Minnesota Pitching through immense pain and on two days rest , Sandy Koufax throws a 3 hit shutout to win the Series and World Series MVP . The home team won all games of the series , except for game seven . 1967 St. Louis Cardinals Boston Red Sox 7 -- 2 Fenway Park , Boston Bob Gibson wins his third game of the Series and also homers . 1968 Detroit Tigers St. Louis Cardinals 4 -- 1 Busch Stadium II , St. Louis , Missouri Mickey Lolich out - duels Bob Gibson . The Tigers rallied from a 3 - 1 deficit to win the series . First game seven loss by the Cardinals in the World Series . 1971 Pittsburgh Pirates Baltimore Orioles 2 -- 1 Memorial Stadium , Baltimore The home team won all games of the series , except for game seven . 1972 Oakland Athletics Cincinnati Reds 3 -- 2 Riverfront Stadium , Cincinnati The A 's first championship in Oakland . The A 's were the last American League team to win a World Series game seven on the road until 2017 . 1973 Oakland Athletics New York Mets 5 -- 2 Oakland -- Alameda County Coliseum , Oakland , California Bert Campaneris and Reggie Jackson hit the only Oakland home runs in this game to set the tone for victory and the series . Cincinnati Reds Boston Red Sox 4 -- 3 Fenway Park , Boston One night after Carlton Fisk 's famous home run , the Reds rebound to take the Series . 1979 Pittsburgh Pirates Baltimore Orioles 4 -- 1 Memorial Stadium , Baltimore The Pittsburgh Pirates , powered by NLCS and World Series MVP Willie Stargell , rally from a 3 - 1 deficit to win the series . The Pirates were the last team to win a road game seven until 2014 . 1982 St. Louis Cardinals Milwaukee Brewers 6 -- 3 Busch Stadium II , St. Louis , Missouri Joaquín Andújar earns his second win of the Series and Bruce Sutter his second save . 1985 Kansas City Royals St. Louis Cardinals 11 -- 0 Royals Stadium , Kansas City , Missouri Second World Series held entirely in the state of Missouri . The Royals also won the 1985 American League Championship Series in seven games . Royals became first team to win the World Series after losing the first two games at home . The Royals were the last team to come back from 3 -- 1 down to win a title , in any of the five major men 's professional sports leagues , until the Cleveland Cavaliers won the 2016 NBA Finals after trailing 3 -- 1 . The 1985 Royals were the last team to win a World Series after trailing 3 -- 1 until the Cubs did so in 2016 . 1986 New York Mets Boston Red Sox 8 -- 5 Shea Stadium , Flushing , Queens , New York Mets become the first team to be one strike away from elimination and come back to win the series . They also become the second team to win the series after losing the first two at home . Minnesota Twins St. Louis Cardinals 4 -- 2 Hubert H. Humphrey Metrodome , Minneapolis The home team won all seven games of the series for the first time in MLB history . 1991 Minnesota Twins Atlanta Braves 1 -- 0 F / 10 Hubert H. Humphrey Metrodome , Minneapolis Like in 1987 , the home team won all games of the series . The Twins won in the bottom of the 10th on a walk - off single by Gene Larkin scoring Dan Gladden after ​ 9 ⁄ scoreless innings of play . Jack Morris goes all ten innings for the Twins and is named the Series MVP . Twins Centerfielder Kirby Puckett hit a famous walk - off home run in the bottom of the eleventh inning of the previous night 's game to force this decisive game . 1997 Florida Marlins Cleveland Indians 3 -- 2 F / 11 Pro Player Stadium , Miami Gardens , Florida Florida becomes the first wild card team to win the World Series . The Marlins rallied for one run in the bottom of the ninth to tie and won in the eleventh on an Édgar Rentería RBI single off Charles Nagy . 2001 Arizona Diamondbacks New York Yankees 3 -- 2 Bank One Ballpark , Phoenix , Arizona Arizona becomes the fastest expansion team to win the World Series , rallying for two runs in the bottom of the ninth for a come - from - behind win , capped off by Luis Gonzalez hitting the winning RBI single off closer Mariano Rivera . The home team won every game of the series . 2002 Anaheim Angels San Francisco Giants 4 -- 1 Edison International Field , Anaheim , California Both World Series teams were wild card entrants into the playoffs . John Lackey became the second rookie pitcher to win a World Series game seven . 2011 St. Louis Cardinals Texas Rangers 6 -- 2 Busch Stadium , St. Louis , Missouri Cardinals became the second team to be one strike away from elimination and come back to win the series , which they did twice . Just like the 1991 series , a walk - off home run in the bottom of the eleventh inning of game six , forced this decisive game with David Freese providing the fire power . Most recent game seven won by the home team . 2014 San Francisco Giants Kansas City Royals 3 -- 2 Kauffman Stadium , Kansas City , Missouri Like in 2002 , both World Series teams were wild card entrants into the playoffs . The Giants became the first NL team to win a game seven on the road since 1979 . NLCS MVP and eventual World Series MVP Madison Bumgarner , normally a starter , pitched five scoreless innings of relief on two days rest to save the Giants ' third title in five years . 2016 Chicago Cubs Cleveland Indians 8 -- 7 F / 10 Progressive Field , Cleveland , Ohio Chicago Cubs ' first World Series win since 1908 . Game played between the two teams ( Cleveland Indians and Chicago Cubs ) with the longest active championship droughts . The Cubs are the first Major League Baseball team to come back to win the World Series from a 3 -- 1 series deficit since the 1985 Royals . Most recent World Series game seven to be decided in extra innings . First extra innings game seven to be won by the road team . 2017 Houston Astros Los Angeles Dodgers 5 -- 1 Dodger Stadium , Los Angeles First instance of back - to - back game sevens since 2001 -- 2002 . First World Series championship for the Astros , who became the first AL team to win a World Series game 7 on the road since 1972 . The Astros became the second team to win two winner - take - all game sevens in the same postseason since the Kansas City Royals in 1985 .
  Fact 0.0: 1909 World Series: Pittsburgh Pirates beat Detroit Tigers 8-0 at Bennett Park, Detroit.
  Fact 0.1: 1909 World Series: Rookie Babe Adams tossed a six-hit shutout for his third win of the series.
  Fact 0.2: 1912 World Series: Boston Red Sox beat New York Giants 3-2 in 10 innings (F/10) at Fenway Park, Boston.
  Fact 0.3: 1912 World Series: The decisive contest was actually game eight, as game two had ended in a tie.
  Fact 0.4: 1924 World Series: Washington Senators beat New York Giants 4-3 in 12 innings (F/12) at Griffith Stadium, Washington, D.C.
  Fact 0.5: 1924 World Series: It was the longest game seven in innings.
  Fact 0.6: 1924 World Series: In the 12th inning, Muddy Ruel doubled after a dropped foul popfly by Giants catcher Hank Gowdy.
  Fact 0.7: 1924 World Series: Muddy Ruel scored on a game-winning RBI double by Earl McNeely.
  Fact 0.8: 1925 World Series: Pittsburgh Pirates beat Washington Senators 9-7 at Forbes Field, Pittsburgh.
  Fact 0.9: 1925 World Series: Pittsburgh became the first team in a best-of-seven series to win when trailing three games to one.
  Fact 0.10: 1926 World Series: St. Louis Cardinals beat New York Yankees 3-2 at Yankee Stadium, the Bronx, New York City.
  Fact 0.11: 1926 World Series: Grover Cleveland Alexander struck out Tony Lazzeri with the bases loaded to end the seventh inning and preserve the Cardinals lead.
  Fact 0.12: 1926 World Series: Two innings later, Babe Ruth was caught stealing to end the World Series.
  Fact 0.13: 1931 World Series: St. Louis Cardinals beat Philadelphia Athletics 4-2 at St. Louis, Missouri.
  Fact 0.14: 1931 World Series: It was the last postseason game for the Athletics while based in Philadelphia.
  Fact 0.15: 1934 World Series: St. Louis Cardinals beat Detroit Tigers 11-0 at Navin Field, Detroit.
  Fact 0.16: 1934 World Series: Dizzy Dean's second win, a six-hit shutout on one day of rest, gave the Cardinals the title.
  Fact 0.17: 1940 World Series: Cincinnati Reds beat Detroit Tigers 2-1 at Crosley Field, Cincinnati.
  Fact 0.18: 1940 World Series: The game was over in a snappy 1:47.
  Fact 0.19: 1940 World Series: A sacrifice bunt and a sacrifice fly scored the winning runs.
  Fact 0.20: 1945 World Series: Detroit Tigers beat Chicago Cubs 9-3 at Wrigley Field, Chicago.
  Fact 0.21: 1945 World Series: It was the Cubs' last World Series appearance until 2016.
  Fact 0.22: 1945 World Series: The row has a see-also note for the Curse of the Billy Goat.
  Fact 0.23: 1946 World Series: St. Louis Cardinals beat Boston Red Sox 4-3 at Sportsman's Park III, St. Louis, Missouri.
  Fact 0.24: 1946 World Series: Enos Slaughter's famous mad dash scored the deciding run to win the game and the series.
  Fact 0.25: 1947 World Series: New York Yankees beat Brooklyn Dodgers 5-2 at Yankee Stadium, the Bronx, New York City.
  Fact 0.26: 1947 World Series: This was the Series that featured Cookie Lavagetto's pinch-hit that spoiled Bill Bevens' no-hit bid.
  Fact 0.27: 1947 World Series: Al Gionfriddo's catch robbed Joe DiMaggio of a home run.
  Fact 0.28: 1952 World Series: New York Yankees beat Brooklyn Dodgers 4-2 at Ebbets Field, Brooklyn, New York City.
  Fact 0.29: 1952 World Series: Billy Martin's game-saving infield catch preserved the Yankees lead and helped win the game and the series.
  Fact 0.30: 1955 World Series: Brooklyn Dodgers beat New York Yankees 2-0 at Yankee Stadium, the Bronx, New York City.
  Fact 0.31: 1955 World Series: It was the Dodgers' only championship in Brooklyn.
  Fact 0.32: 1955 World Series: The home team won all games of the series except game seven.
  Fact 0.33: 1956 World Series: New York Yankees beat Brooklyn Dodgers 9-0 at Ebbets Field, Brooklyn, New York City.
  Fact 0.34: 1956 World Series: It was Jackie Robinson's final Major League game.
  Fact 0.35: 1956 World Series: It was the last postseason game at Ebbets Field.
  Fact 0.36: 1956 World Series: It was the Dodgers' last postseason game before moving to Los Angeles.
  Fact 0.37: 1956 World Series: The home team won all games of the series except game seven.
  Fact 0.38: 1957 World Series: Milwaukee Braves beat New York Yankees 5-0 at Yankee Stadium, the Bronx, New York City.
  Fact 0.39: 1957 World Series: It was the Braves' only championship in Milwaukee.
  Fact 0.40: 1958 World Series: New York Yankees beat Milwaukee Braves 6-2 at County Stadium, Milwaukee.
  Fact 0.41: 1958 World Series: The Yankees won their 18th World Series title after being down three games to one.
  Fact 0.42: 1958 World Series: It was the Braves' last postseason game in Milwaukee.
  Fact 0.43: 1960 World Series: Pittsburgh Pirates beat New York Yankees 10-9 at Forbes Field, Pittsburgh.
  Fact 0.44: 1960 World Series: Bill Mazeroski hit the walk-off home run to win the Pirates the championship.
  Fact 0.45: 1960 World Series: It was the last postseason game in Forbes Field.
  Fact 0.46: 1960 World Series: It was the last postseason game for Yankees' manager Casey Stengel.
  Fact 0.47: 1960 World Series: It was the highest scoring game seven.
  Fact 0.48: 1962 World Series: New York Yankees beat San Francisco Giants 1-0 at Candlestick Park, San Francisco.
  Fact 0.49: 1962 World Series: With runners on second and third and two outs, Willie McCovey's line drive to Bobby Richardson ended the game.
  Fact 0.50: 1962 World Series: It was the first game seven that ended with a 1-0 score.
  Fact 0.51: 1964 World Series: St. Louis Cardinals beat New York Yankees 7-5 at Busch Stadium I, St. Louis, Missouri.
  Fact 0.52: 1964 World Series: It was the Yankees' last World Series appearance until 1976.
  Fact 0.53: 1964 World Series: It was the last postseason game at Busch Stadium I.
  Fact 0.54: 1965 World Series: Los Angeles Dodgers beat Minnesota Twins 2-0 at Metropolitan Stadium, Bloomington, Minnesota.
  Fact 0.55: 1965 World Series: Sandy Koufax pitched through immense pain and on two days rest.
  Fact 0.56: 1965 World Series: Sandy Koufax threw a 3-hit shutout to win the Series.
  Fact 0.57: 1965 World Series: Sandy Koufax was named World Series MVP.
  Fact 0.58: 1965 World Series: The home team won all games of the series except game seven.
  Fact 0.59: 1967 World Series: St. Louis Cardinals beat Boston Red Sox 7-2 at Fenway Park, Boston.
  Fact 0.60: 1967 World Series: Bob Gibson wins his third game of the Series.
  Fact 0.61: 1967 World Series: Bob Gibson also homers in this game.
  Fact 0.62: 1968 World Series: Detroit Tigers beat St. Louis Cardinals 4-1 at Busch Stadium II, St. Louis, Missouri.
  Fact 0.63: 1968 World Series: Mickey Lolich out-dueled Bob Gibson.
  Fact 0.64: 1968 World Series: The Tigers rallied from a 3-1 deficit to win the series.
  Fact 0.65: 1968 World Series: It was the first game seven loss by the Cardinals in the World Series.
  Fact 0.66: 1971 World Series: Pittsburgh Pirates beat Baltimore Orioles 2-1 at Memorial Stadium, Baltimore.
  Fact 0.67: 1971 World Series: The home team won all games of the series except game seven.
  Fact 0.68: 1972 World Series: Oakland Athletics beat Cincinnati Reds 3-2 at Riverfront Stadium, Cincinnati.
  Fact 0.69: 1972 World Series: It was the A's first championship in Oakland.
  Fact 0.70: 1972 World Series: The A's were the last American League team to win a World Series game seven on the road until 2017.
  Fact 0.71: 1973 World Series: Oakland Athletics beat New York Mets 5-2 at Oakland-Alameda County Coliseum, Oakland, California.
  Fact 0.72: 1973 World Series: Bert Campaneris and Reggie Jackson hit the only Oakland home runs in this game.
  Fact 0.73: 1973 World Series: Those home runs set the tone for victory and the series.
  Fact 0.74: A row with no year listed: Cincinnati Reds beat Boston Red Sox 4-3 at Fenway Park, Boston.
  Fact 0.75: A row with no year listed: It was one night after Carlton Fisk's famous home run.
  Fact 0.76: A row with no year listed: The Reds rebounded to take the Series.
  Fact 0.77: 1979 World Series: Pittsburgh Pirates beat Baltimore Orioles 4-1 at Memorial Stadium, Baltimore.
  Fact 0.78: 1979 World Series: The Pirates were powered by NLCS and World Series MVP Willie Stargell.
  Fact 0.79: 1979 World Series: The Pirates rallied from a 3-1 deficit to win the series.
  Fact 0.80: 1979 World Series: The Pirates were the last team to win a road game seven until 2014.
  Fact 0.81: 1982 World Series: St. Louis Cardinals beat Milwaukee Brewers 6-3 at Busch Stadium II, St. Louis, Missouri.
  Fact 0.82: 1982 World Series: Joaquín Andújar earned his second win of the Series.
  Fact 0.83: 1982 World Series: Bruce Sutter earned his second save.
  Fact 0.84: 1985 World Series: Kansas City Royals beat St. Louis Cardinals 11-0 at Royals Stadium, Kansas City, Missouri.
  Fact 0.85: 1985 World Series: It was the second World Series held entirely in the state of Missouri.
  Fact 0.86: 1985 Royals also won the 1985 American League Championship Series in seven games.
  Fact 0.87: 1985 Royals became the first team to win the World Series after losing the first two games at home.
  Fact 0.88: 1985 Royals were the last team to come back from 3-1 down to win a title in any of the five major men's professional sports leagues until the Cleveland Cavaliers won the 2016 NBA Finals after trailing 3-1.
  Fact 0.89: 1985 Royals were the last team to win a World Series after trailing 3-1 until the Cubs did so in 2016.
  Fact 0.90: 1986 World Series: New York Mets beat Boston Red Sox 8-5 at Shea Stadium, Flushing, Queens, New York.
  Fact 0.91: 1986 World Series: The Mets became the first team to be one strike away from elimination and come back to win the series.
  Fact 0.92: 1986 World Series: The Mets became the second team to win the series after losing the first two games at home.
  Fact 0.93: A row with no year listed: Minnesota Twins beat St. Louis Cardinals 4-2 at Hubert H. Humphrey Metrodome, Minneapolis.
  Fact 0.94: A row with no year listed: The home team won all seven games of the series for the first time in MLB history.
  Fact 0.95: 1991 World Series: Minnesota Twins beat Atlanta Braves 1-0 in 10 innings (F/10) at Hubert H. Humphrey Metrodome, Minneapolis.
  Fact 0.96: 1991 World Series: Like in 1987, the home team won all games of the series.
  Fact 0.97: 1991 World Series: The Twins won in the bottom of the 10th on a walk-off single by Gene Larkin scoring Dan Gladden after 9 and a fraction scoreless innings of play.
  Fact 0.98: 1991 World Series: Jack Morris went all ten innings for the Twins and was named Series MVP.
  Fact 0.99: 1991 World Series: Twins centerfielder Kirby Puckett hit a famous walk-off home run in the bottom of the eleventh inning of the previous night's game to force this decisive game.
  Fact 0.100: 1997 World Series: Florida Marlins beat Cleveland Indians 3-2 in 11 innings (F/11) at Pro Player Stadium, Miami Gardens, Florida.
  Fact 0.101: 1997 World Series: Florida became the first wild card team to win the World Series.
  Fact 0.102: 1997 World Series: The Marlins rallied for one run in the bottom of the ninth to tie.
  Fact 0.103: 1997 World Series: The Marlins won in the eleventh on an Édgar Rentería RBI single off Charles Nagy.
  Fact 0.104: 2001 World Series: Arizona Diamondbacks beat New York Yankees 3-2 at Bank One Ballpark, Phoenix, Arizona.
  Fact 0.105: 2001 World Series: Arizona became the fastest expansion team to win the World Series.
  Fact 0.106: 2001 World Series: Arizona rallied for two runs in the bottom of the ninth for a come-from-behind win.
  Fact 0.107: 2001 World Series: Luis Gonzalez hit the winning RBI single off closer Mariano Rivera.
  Fact 0.108: 2001 World Series: The home team won every game of the series.
  Fact 0.109: 2002 World Series: Anaheim Angels beat San Francisco Giants 4-1 at Edison International Field, Anaheim, California.
  Fact 0.110: 2002 World Series: Both World Series teams were wild card entrants into the playoffs.
  Fact 0.111: 2002 World Series: John Lackey became the second rookie pitcher to win a World Series game seven.
  Fact 0.112: 2011 World Series: St. Louis Cardinals beat Texas Rangers 6-2 at Busch Stadium, St. Louis, Missouri.
  Fact 0.113: 2011 World Series: The Cardinals became the second team to be one strike away from elimination and come back to win the series.
  Fact 0.114: 2011 World Series: The Cardinals did come back to win the series twice.
  Fact 0.115: 2011 World Series: Just like the 1991 series, a walk-off home run in the bottom of the eleventh inning of game six forced this decisive game.
  Fact 0.116: 2011 World Series: David Freese provided the firepower in that game six walk-off home run.
  Fact 0.117: 2011 World Series: The most recent game seven was won by the home team.
  Fact 0.118: 2014 World Series: San Francisco Giants beat Kansas City Royals 3-2 at Kauffman Stadium, Kansas City, Missouri.
  Fact 0.119: 2014 World Series: Like in 2002, both World Series teams were wild card entrants into the playoffs.
  Fact 0.120: 2014 World Series: The Giants became the first NL team to win a game seven on the road since 1979.
  Fact 0.121: 2014 World Series: Madison Bumgarner was the NLCS MVP and eventual World Series MVP.
  Fact 0.122: 2014 World Series: Madison Bumgarner is normally a starter.
  Fact 0.123: 2014 World Series: Madison Bumgarner pitched five scoreless innings of relief on two days rest.
  Fact 0.124: 2014 World Series: Madison Bumgarner's relief pitching saved the Giants' third title in five years.
  Fact 0.125: 2016 World Series: Chicago Cubs beat Cleveland Indians 8-7 in 10 innings (F/10) at Progressive Field, Cleveland, Ohio.
  Fact 0.126: 2016 World Series: It was the Cubs' first World Series win since 1908.
  Fact 0.127: 2016 World Series: The game was played between the two teams with the longest active championship droughts.
  Fact 0.128: 2016 World Series: The Cubs are the first Major League Baseball team to come back to win the World Series from a 3-1 series deficit since the 1985 Royals.
  Fact 0.129: 2016 World Series: It was the most recent World Series game seven to be decided in extra innings.
  Fact 0.130: 2016 World Series: It was the first extra innings game seven to be won by the road team.
  Fact 0.131: 2017 World Series: Houston Astros beat Los Angeles Dodgers 5-1 at Dodger Stadium, Los Angeles.
  Fact 0.132: 2017 World Series: It was the first instance of back-to-back game sevens since 2001-2002.
  Fact 0.133: 2017 World Series: It was the first World Series championship for the Astros.
  Fact 0.134: 2017 World Series: The Astros became the first AL team to win a World Series game 7 on the road since 1972.
  Fact 0.135: 2017 World Series: The Astros became the second team to win two winner-take-all game sevens in the same postseason since the 1985 Kansas City Royals.

Retrieved items:
<item 1>
A game seven is the decisive game in a best-of-seven playoff series, akin to a championship final; most football codes do not use series. MLB game sevens occur
in League Championship Series and World Series, not the best-of-five Division Series. The World Series has had 39 decisive game sevens; home teams are 19–20.
Four early World Series used best-of-nine formats, but none reached game nine. In extra-inning game sevens, the Giants and Indians lost both
</item 1>

<item 2>
Game seven
Game played between the two teams ( Cleveland Indians and Chicago Cubs ) with the longest active championship droughts . The Cubs are the first Major League
Baseball team to come back to win the World Series from a 3 -- 1 series deficit since the 1985 Royals . Most recent World Series game seven to be decided in
extra innings . First extra innings game seven to be won by the road team . 2017 Houston Astros Los Angeles Dodgers 5 -- 1 Dodger Stadium , Los Angeles First
instance of back - to - back game sevens since 2001 -- 2002 . First World Series championship for the Astros , who became the first AL team to win a World
Series game 7 on the road since 1972 . The Astros became the second team to win two winner - take - all game sevens in the same postseason since the Kansas City
Royals in 1985 . League Championship Series ( edit ) The League Championship Series were originally decided best of five from 1969 until 1984 . Since 1985 when
both were expanded to best of seven contests , 16 series ( 7 American , 9 National ) were decided in game seven .
</item 2>

<item 3>
A Game 7 is the deciding contest of a tied best-of-seven series, common in MLB, NBA and NHL playoffs. World Series Game 7s have seen home teams go 19–20;
highlights include the 2016 Cubs’ comeback from 3–1, the first road-team extra-inning win, and 1988’s unprecedented game featuring Canseco’s grand slam and
Gibson’s walk-off homer. Other memorable moments include Slaughter’s 1946 “Mad Dash,”
</item 3>

<item 4>
The 2016 World Series Game 7 matched Cleveland and Chicago, MLB’s teams with the longest active championship droughts. The Cubs overcame a 3–1 deficit, becoming
the first team to do so since the 1985 Royals; they won the latest extra-inning World Series Game 7 and the first such game won by the road team. In 2017,
Houston beat the Dodgers 5–1 for its first title, in the first back-to-back World Series Game
</item 4>

<item 5>
In World Series Game Sevens, home teams have a 19–20 record. Notable results: Pittsburgh beat Detroit 8–0 in 1909 as rookie Babe Adams threw a six-hit shutout;
Boston beat New York 3–2 in 10 innings in 1912, in the series’ eighth game after a tie; Washington won 4–3 in 12 innings in 1924, the longest Game Seven, on
Earl McNeely’s RBI double; and
</item 5>

<item 6>
2017 World Series
Game 5 lasted five hours and seventeen minutes , making it the second longest World Series game in history by time , trailing only the 14 inning contest between
the Astros and the Chicago White Sox in Game 3 of the 2005 World Series in the same ballpark , as well as the third World Series game ever to go beyond five
hours of game time . This game is considered one of the greatest World Series games of all time . Game 6 ( edit ) Kenley Jansen recorded a two - inning save in
Game 6 .
</item 6>

<item 7>
The 2017 World Series Game 5 lasted 5 hours, 17 minutes—the second-longest by time, behind the 2005 Astros–White Sox Game 3 at the same ballpark—and is
considered one of the greatest World Series games. In Game 6, Kenley Jansen earned a two-inning save. In 2005, Houston disputed a roof-closure ruling, citing
its stronger record with the roof closed; officials said the decision followed established rules and weather forecasts.
</item 7>

<item 8>
Game seven
1 World Series 1.1. 2 League Championship Series 1.2 National Hockey League 1.3 National Basketball Association 1.4 Chinese Professional Baseball League 2
Comebacks 3 References Examples ( edit ) Major League Baseball ( edit ) In the MLB Postseason , game seven can occur in the League Championship Series and the
World Series ( the divisional series are best of five ) . In the World Series , there have been 39 decisive game sevens ( of the four World Series that were
contested on a best - of - nine basis , none went to a game nine , though all four included a non-decisive game seven ) . World Series ( edit ) World Series
that were decided on a game seven : All extra inning game sevens involve three franchises each with two appearances , the New York / San Francisco Giants (
losing both ) , the Washington Senators / Minnesota Twins ( winning both ) and the Cleveland Indians ( losing both ) . Home teams are 19 -- 20 in game sevens .
</item 8>

<item 9>
Game seven
2012 NLCS San Francisco Giants St. Louis Cardinals 9 -- 0 AT&T Park , San Francisco , California First Giants victory in a winner - take - all game seven . The
Giants would go on to sweep the Detroit Tigers in the World Series . 2017 ALCS Houston Astros New York Yankees 4 -- 0 Minute Maid Park , Houston , Texas The
home team won all games of the series . Houston became the first franchise to win the pennant in both the National League and American League . The Astros would
go on to win the World Series over the Los Angeles Dodgers in seven games , becoming the second MLB team to win two game sevens in a single postseason .
National Hockey League ( edit ) Main article : List of NHL game sevens In the NHL playoffs , game seven can occur in all playoff series . In the Stanley Cup
Final , the NHL gained complete control of the Cup in 1926 , and the league instituted the best - of - seven series starting in the 1938 -- 39 season . No
Stanley Cup Final game seven has ever ended with a 1 -- 0 score .
</item 9>

<item 10>
Game seven
Louis the NL Pennant . The Cardinals would lose the World Series as the Minnesota Twins give new meaning to the term home - field advantage in the World Series
as only the decisive game seven is close . 1988 NLCS Los Angeles Dodgers New York Mets 6 -- 0 Dodger Stadium , Los Angeles Orel Hershiser pitches a five - hit
shutout to send the Dodgers to the World Series . Until 2017 , this was the only game seven of any postseason series to be played in Dodger Stadium . 1991 NLCS
Atlanta Braves Pittsburgh Pirates 4 -- 0 Three Rivers Stadium , Pittsburgh , Pennsylvania Back - to - back shutouts in games six and seven helps the Braves win
the pennant . The Braves would lose the series to Minnesota in seven games in what is regarded as the best World Series of all time . 1992 NLCS Atlanta Braves
Pittsburgh Pirates 3 -- 2 Atlanta -- Fulton County Stadium , Atlanta Atlanta scored all three runs in the bottom of the ninth . Pinch hitter Francisco Cabrera
drove in the two winning runs with a two - out single with the bases loaded . The winning run was scored by Sid Bream .
</item 10>
=== END ITEM id=cbaf3b8d27603b0a ===
