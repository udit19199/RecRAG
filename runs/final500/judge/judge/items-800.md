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

=== ITEM id=a4d403d844b753c3 ===
Gold sentences and facts:
Sentence 0: In most of the world speed limit signs display the limitation within a red circle . This design follows the style set out by the Vienna Convention on Road Signs and Signals , with the exception of United States and Canada . All countries globally worldwide ( except the United Kingdom and United States ) use metric speed limits in kilometres per hour . Metric signs are optional in the United States though less commonly seen , while Samoa uses both units . In Ontario , Canada , the type , location , and frequency of speed limit signs is covered by regulation 615 of the Ontario Highway Traffic Act .
  Fact 0.0: In most of the world, speed limit signs display the limitation within a red circle.
  Fact 0.1: The red-circle speed limit sign design follows the style set out by the Vienna Convention on Road Signs and Signals.
  Fact 0.2: The Vienna Convention style is not followed by the United States and Canada.
  Fact 0.3: All countries worldwide, except the United Kingdom and the United States, use metric speed limits in kilometres per hour.
  Fact 0.4: Metric speed limit signs are optional in the United States.
  Fact 0.5: Metric speed limit signs are less commonly seen in the United States.
  Fact 0.6: Samoa uses both units for speed limits.
  Fact 0.7: In Ontario, Canada, the type, location, and frequency of speed limit signs is covered by regulation 615 of the Ontario Highway Traffic Act.

Retrieved items:
<item 1>
Speed limit
p. 343 . External links ( edit ) Wikimedia Commons has media related to Speed limit signs and Driver speed and relative risk of an accident . Global map of
speed limits from Wikispeedia A Walk Through The History of Speed Limit Signs in The US John F. Carr 's State Traffic and Speed Laws Global map of speed limits
from OpenStreetMap Traffic signs Signs Warning sign Regulatory sign Prohibitory traffic sign Mandatory sign Special regulation sign Stop sign Yield sign
Crossbuck Dead End sign One - way traffic Speed limit ( by country ) Advisory speed limit Information sign Direction , position , or indication sign Variable -
message sign Bilingual sign Driver location signs Logo sign Priority to the right By comparison Comparison of European road signs Comparison of MUTCD -
influenced traffic signs Comparison of traffic signs in English - speaking countries By country Africa Botswana Mauritius South Africa Zimbabwe Asia Azerbaijan
Bangladesh Brunei Cambodia China Hong Kong India Indonesia Iran Israel Japan Laos Malaysia Nepal Pakistan Singapore South Korea Sri Lanka Thailand Turkey United
Arab Emirates Vietnam Europe Austria Bosnia and Herzegovina Czech Republic Denmark Estonia Finland France Germany Greece Iceland Ireland Italy Liechtenstein
Netherlands Norway Poland Sweden Switzerland Ukraine United Kingdom Scotland Wales North America Canada Mexico United States Oceania Australia New Zealand South
America Argentina Brazil Chile Colombia Lights Traffic
</item 1>

<item 2>
Speed limit
The U.S. Federal Highway Administration 's Manual on Uniform Traffic Control Devices provides guidelines for the appearance of speed limit signs . In Alaska and
California , speed limits are often labeled `` MAXIMUM SPEED '' instead . In Oregon , most speed limit signs just say `` SPEED '' and the number . Canada has
similar signs bearing the legend `` MAXIMUM '' and in km / h instead of mph . `` Maximum '' is used instead of `` Speed Limit '' because it has similar meanings
in English and French . Australian signs are rectangular but have a red circle like the Vienna Convention signs . The speed limit is commonly set at or below
the 85th percentile operating speed ( being the speed which no more than 15 % of traffic is exceeding ) and in the US is frequently set 4 to 8 mph ( 6 to 13 km
/ h ) below that speed . Thus , if the 85th percentile operating speed as measured by a Traffic and Engineering Survey exceeds the design speed , legal
protection is given to motorists traveling at such speeds ( design speed is `` based on conservative assumptions about driver , vehicle and roadway
characteristics '' ) .
</item 2>

<item 3>
Speed limit
In international European roads , speed should be taken into account at design stage . Maximum speed recommended by international European road design standard
Road classification 60 km / h 80 km / h ( 50 mph ) 100 km / h ( 60 mph ) 120 km / h ( 75 mph ) 140 km / h ( 85 mph ) Motorway x 80 100 120 140 Express road 60
80 100 120 x Road 60 80 100 x x Minimum speed limits ( edit ) UK minimum speed limit sign Some roads also have `` minimum speed limits '' , where slow speeds
can impede traffic flow or be dangerous . Signs often use blue circles based on the obligatory signs of the Vienna Convention on Road Signs and Signals . A
Japanese minimum speed sign has the same design as a maximum speed sign , but with a horizontal line below the number . In the United States of America , they
are also identical to their respective maximum speed limit signs with the exception of the text `` MINIMUM SPEED '' . This minimum speed is not so common , as
the risk usually raise with higher speed and not with lower speed .
</item 3>

<item 4>
Speed limit
In the European Union , large signposts showing the national ( default ) speed limits of the respective country are usually erected immediately after border
crossings , with a repeater sign some 200 to 500 m ( 660 to 1,640 ft ) after the first sign . Some places provide an additional `` speed zone ahead '' ahead of
the restriction and speed limit reminder signs may appear at regular intervals which may be painted on the road surface . Signs are normally placed on both
sides of the road and in some places there are small ( less than 1 / 4 the size of the sign ) rectangular orange reflector flags attached to both upper right
corners of both signs . The speed limit sign marking the new speed zone may also have the orange flags ; this practice can be observed in New York on highways
where the speed limit varies such as New York State Route 17 . Speed limits on European Motorways in km / h Signage in many countries , especially in Europe ,
conforms to the Vienna Convention on Road Signs and Signals using black text with a red circle on a white background . In the United States , the signs are
usually rectangular with the words `` SPEED LIMIT '' and the values in black on a white background .
</item 4>

<item 5>
The references on speed limits cite MUTCD Sections 2B.13–2B.16; California Vehicle Code §21400(b), dated January 1, 2012; and NCHRP Report 504, *Design Speed,
Operating Speed, and Posted Speed Practices*, published by the National Academies Press in 2003 (p. 88).
</item 5>

<item 6>
Speed limit
Variable speed limits ( edit ) Example variable speed limit sign in the United States Digital speed limit sign for variable speed limits In 1965 , the first
known experiments with variable speed limit signs took place on a 30 km stretch of German motorway A8 between Munich and the border city of Salzburg , Austria .
Mechanically variable message signs could display speeds of 60 , 80 and 100 km / h , `` danger zone '' or `` accident '' . Personnel monitored traffic using
video technology , and manually controlled the signage . Beginning in the 1970s , additional advanced traffic control systems were put into service . Modern
motorway control systems can work without human intervention using various types of sensors to measure traffic flow and weather conditions . In 2009 , 1,300 km
( 810 mi ) of German motorways were equipped with such systems . In the late 1960s , heavily traveled portions of the New Jersey Turnpike began using variable
speed limit signs , in combination with variable message signs . Officials can adjust the speed limit according to weather , traffic conditions , and
construction .
</item 6>

<item 7>
European speed-limit signs typically show national defaults after borders, with repeaters and reminders; EU signs commonly use black numerals in a red circle on
white, while U.S. signs are usually rectangular, with regional wording variations. Canada uses “MAXIMUM” and km/h; Australia combines a rectangle with a red
circle. Limits often reflect the 85th-percentile traffic speed, sometimes set 4–8 mph lower in the U.S. Design speed guides road geometry but does not
necessarily define a
</item 7>

<item 8>
Speed limit
The speeding fatality rate for local roads is three times that for Interstates Jump up ^ http://www.pe.com/articles/speed-776756-survey-limit.html Jump up ^
http://caselaw.findlaw.com/mt-supreme-court/1110919.html Jump up ^
http://vincenzeslaw.com/4-ways-virginia-reckless-driving-charges-handled-differently-fairfax-county-arlington-alexandria-stafford/ Jump up ^ `` Traffic
Operations : Road Signs '' . Government of South Australia . 2002 - 08 - 23 ( last update date ) . Retrieved 2010 - 09 - 26 . Most regulatory signs are
rectangular with a black legend on a white background ... ( exceptions include ) speed restriction signs ( symbol enclosed in a red circle ) '' Check date
values in : date = ( help ) Jump up ^ `` Speed Zoning Information '' ( PDF ) . Institute of Transportation Engineers . 22 March 2004 . Retrieved 2009 - 09 - 17
. Jump up ^ MUTCD Sections 2B.
</item 8>

<item 9>
Speed limit
Per distance traveled , consequences of inappropriate speed are more frequent on lower speed , lower quality roads ; in the United States , for example , the ``
speeding fatality rate for local roads is three times that for Interstates '' Citations for violations of the basic speed law without a crash have sometimes
been ruled unfairly vague or arbitrary , hence a violation the due process of law , at least in the State of Montana . Even within states , differing
jurisdictions ( counties and cities ) choose to prosecute similar cases with differing approaches . Maximum speed limits ( edit ) A speed limit sign in Ontario
In France city sign has value of speed limit A street limited at 30 km / h End of speed limit , previous speed limit applies Most public roads in most countries
have a legally assigned numerical maximum speed limit which applies on all roads unless otherwise stated ; lower speed limits are often shown on a sign at the
start of the restricted section , although the presence of streetlights or the physical arrangement of the road may sometimes also be used instead . A posted
speed limit may only apply to that road or to all roads beyond the sign that defines them depending on local laws .
</item 9>

<item 10>
Posted speed limits do not remove drivers’ duty to travel at a safe speed. French law requires slowing in situations such as poor visibility, bends, hazards,
pedestrians, cyclists, animals, and busy or narrow roads; US federal rules and state “basic speed laws,” such as California’s, similarly require reasonable
speeds, even below the posted limit. Weather, road conditions, traffic, and limited visibility can make the safe speed lower. Speed-related fatalities are more
common on lower-quality roads. Enforcement of broad
</item 10>
=== END ITEM id=a4d403d844b753c3 ===

=== ITEM id=07911ebf82e2a41a ===
Gold sentences and facts:
Sentence 0: Starting at 500 million Population ( in billions ) 0.5 8 Year 1500 1804 1927 2024 Years elapsed 304 123 47 50
  Fact 0.0: The starting value is 500 million.
  Fact 0.1: The population values, in billions, are 0.5 and 8.
  Fact 0.2: The year values are 1500, 1804, 1927 and 2024.
  Fact 0.3: The years elapsed values are 304, 123, 47 and 50.

Retrieved items:
<item 1>
World population
Projected figures vary depending on underlying statistical assumptions and the variables used in projection calculations , especially the fertility variable .
Long - range predictions to 2150 range from a population decline to 3.2 billion in the `` low scenario '' , to `` high scenarios '' of 24.8 billion . One
extreme scenario predicted a massive increase to 256 billion by 2150 , assuming the global fertility rate remained at its 1995 level of 3.04 children per woman
; however , by 2010 the global fertility rate had declined to 2.52 . There is no estimation for the exact day or month the world 's population surpassed one or
two billion . The points at which it reached three and four billion were not officially noted , but the International Database of the United States Census
Bureau placed them in July 1959 and April 1974 respectively . The United Nations did determine , and commemorate , the `` Day of 5 Billion '' on July 11 , 1987
, and the `` Day of 6 Billion '' on October 12 , 1999 . The Population Division of the United Nations declared the `` Day of 7 Billion '' to be October 31 ,
2011 .
</item 1>

<item 2>
World population
The vertical axis is logarithmic and is in millions of people . Population size fluctuates at differing rates in differing regions . Nonetheless , population
growth is the long - standing trend on all inhabited continents , as well as in most individual states . During the 20th century , the global population saw its
greatest increase in known history , rising from about 1.6 billion in 1900 to over 6 billion in 2000 . A number of factors contributed to this increase ,
including the lessening of the mortality rate in many countries by improved sanitation and medical advances , and a massive increase in agricultural
productivity attributed to the Green Revolution . In 2000 , the United Nations estimated that the world 's population was growing at an annual rate of 1.14 % (
equivalent to around 75 million people ) , down from a peak of 88 million per year in 1989 . By 2000 , there were approximately ten times as many people on
Earth as there had been in 1700 .
</item 2>

<item 3>
World population
The transition from hyperbolic growth to slower rates of growth is related to the demographic transition . According to the Russian demographer Sergey Kapitsa ,
the world population grew between 67,000 BC and 1965 according to the following formula : N = C τ arccot ⁡ T 0 − T τ ( \ displaystyle N = ( \ frac ( C ) ( \ tau
) ) \ operatorname ( arccot ) ( \ frac ( T_ ( 0 ) - T ) ( \ tau ) ) ) where N is current population T is the current year C = ( 1.86 ± 0.01 ) 10 T = 2007 ± 1 τ
( \ displaystyle \ tau ) = 42 ± 1 Years for world population to double ( edit ) According to linear interpolation and extrapolation of UNDESA population
estimates , the world population has doubled , or will double , in the years listed in the tables below ( with two different starting points ) . During the 2nd
millennium , each doubling took roughly half as long as the previous doubling , fitting the hyperbolic growth model mentioned above .
</item 3>

<item 4>
World population
Milestones by the billions ( edit ) Main article : World population milestones World population milestones in billions Population 5 6 7 8 9 Year 1804 1927 1960
1999 2011 2024 2042 Years elapsed -- -- 123 33 14 13 12 12 13 18 It is estimated that the world population reached one billion for the first time in 1804 . It
was another 123 years before it reached two billion in 1927 , but it took only 33 years to reach three billion in 1960 . Thereafter , the global population
reached four billion in 1974 , five billion in 1987 , six billion in 1999 and , according to the United States Census Bureau , seven billion in March 2012 . The
United Nations , however , estimated that the world population reached seven billion in October 2011 . According to current projections , the global population
will reach eight billion by 2024 , and will likely reach around nine billion by 2042 . Alternative scenarios for 2050 range from a low of 7.4 billion to a high
of more than 10.6 billion .
</item 4>

<item 5>
World population
000 BC < 0.015 0 0 10,000 BC 8000 BC 5 6500 BC 5 5000 BC 5 4000 BC 7 3000 BC 14 2000 BC 27 1000 BC 50 7 33 9 500 BC 100 14 66 16 AD 1 200 23 141 28 1000 400 70
269 50 8 1500 458 86 243 84 39 1600 580 114 339 111 10 1700 682 106 436 125 10 1750 791 106 502 163 16 1800 1,000 107 656 203 24 7 1850 1,262 111 809 276 38 26
1900 1,650 133 947 408 74 82 6 1950 2,525 229 1,394 549 169 172 12.7 1955 2,758 254 1,534 577 193 187 14.
</item 5>

<item 6>
Graph query row:
entity: World population
relationship: RELATED_TO
milestone: World population reached one billion
year: 1804
connection: None
</item 6>

<item 7>
World population
Early estimates for the population of the world date to the 17th century : William Petty in 1682 estimated world population at 320 million ( modern estimates
ranging close to twice this number ) ; by the late 18th century , estimates ranged close to one billion ( consistent with modern estimates ) . More refined
estimates , broken down by continents , were published in the first half of the 19th century , at 600 to 1000 million in the early 1800s and at 800 to 1000
million in the 1840s . Antiquity and Middle ages ( edit ) Main articles : Classical demography and Medieval demography Estimates of the population of the world
at the time agriculture emerged in around 10,000 BC have ranged between 1 million and 15 million . Even earlier , genetic evidence suggests humans may have gone
through a population bottleneck of between 1,000 and 10,000 people about 70,000 BC , according to the Toba catastrophe theory . By contrast , it is estimated
that around 50 -- 60 million people lived in the combined eastern and western Roman Empire in the 4th century AD .
</item 7>

<item 8>
World population
758 254 1,534 577 193 187 14.2 1960 3,018 285 1,687 606 221 204 15.8 1965 3,322 322 1,875 635 254 219 17.5 1970 3,682 366 2,120 657 288 231 19.7 4,061 416 2,378
677 326 242 21.5 1980 4,440 478 2,626 694 365 254 23.0 1985 4,853 550 2,897 708 406 267 24.9 5,310 632 3,202 721 447 281 27.0 1995 5,735 720 3,475 728 487 296
29.1 2000 6,127 814 3,714 726 527 314 31.1 2005 6,520 920 3,945 729 564 329 33.4 6,930 1,044 4,
</item 8>

<item 9>
Graph query row:
entity: World population
relationship: RELATED_TO
milestone: World population reached two billion
year: 1927
connection: None
</item 9>

<item 10>
World population
However , after 2024 , it is unlikely that there will be another doubling of the global population in the 21st century . Historic chart showing the periods of
time the world population has taken to double , from 1700 to 2000 Starting at 500 million Population ( in billions ) 0.5 8 Year 1500 1804 1927 2024 Years
elapsed 304 123 47 50 Starting at 375 million Population ( in billions ) 0.375 0.75 1.5 6 Year 1171 1715 1881 1960 1999 Years elapsed 544 166 79 39
Overpopulation ( edit ) Main articles : Human overpopulation and Malthusian catastrophe Predictions of scarcity ( edit ) Greater Los Angeles lies on a coastal
mediterranean savannah with a small watershed that is able to support at most one million people on its own water ; as of 2015 , the area has a population of
over 18 million . Researchers predict that similar cases of resource scarcity will grow more common as the world population increases .
</item 10>
=== END ITEM id=07911ebf82e2a41a ===

=== ITEM id=db78ad82a0383821 ===
Gold sentences and facts:
Sentence 0: Shortwave radio is radio transmission using shortwave radio frequencies . There is no official definition of the band , but the range always includes all of the high frequency band ( HF ) , and generally extends from 1.7 -- 30 MHz ( 176.3 -- 10.0 m ) ; from the high end of the medium frequency band ( MF ) just above the mediumwave AM broadcast band , to the end of the HF band .
  Fact 0.0: Shortwave radio is radio transmission using shortwave radio frequencies.
  Fact 0.1: There is no official definition of the shortwave band.
  Fact 0.2: The shortwave band range always includes the high frequency band (HF).
  Fact 0.3: The shortwave band generally extends from 1.7 to 30 MHz (176.3 to 10.0 m).
  Fact 0.4: The shortwave band extends from the high end of the medium frequency band (MF), just above the mediumwave AM broadcast band.
  Fact 0.5: The shortwave band extends to the end of the HF band.
Sentence 1: Very little infrastructure is required for long - distance two - way communications using shortwave radio . All one needs is a pair of transceivers , each with an antenna , and a source of energy ( such as a battery , a portable generator , or the electrical grid ) . This makes shortwave radio one of the most robust means of communications , which can be disrupted only by interference or bad ionospheric conditions . Modern digital transmission modes such as MFSK and Olivia are even more robust , allowing successful reception of signals well below the noise floor of a conventional receiver .
  Fact 1.0: Very little infrastructure is required for long-distance two-way communication using shortwave radio.
  Fact 1.1: All one needs for shortwave radio communication is a pair of transceivers, each with an antenna.
  Fact 1.2: All one needs for shortwave radio communication is a source of energy, such as a battery, a portable generator, or the electrical grid.
  Fact 1.3: Shortwave radio is one of the most robust means of communications.
  Fact 1.4: Shortwave radio communications can be disrupted only by interference or bad ionospheric conditions.
  Fact 1.5: Modern digital transmission modes such as MFSK and Olivia are even more robust.
  Fact 1.6: Modern digital transmission modes such as MFSK and Olivia allow successful reception of signals well below the noise floor of a conventional receiver.

Retrieved items:
<item 1>
Shortwave radio
The drawback is the receiver is more complicated , since it must re-create the carrier to recover the signal . Small errors in the detection process greatly
affect the pitch of the received signal . As a result , single sideband is not used for music or general broadcast . Single sideband is used for long - range
voice communications by ships and aircraft , Citizen 's Band , and amateur radio operators . Lower sideband ( LSB ) is generally used below 9 MHz and USB (
upper sideband ) above 9 MHz . Vestigal sideband transmits the carrier and one complete sideband , but filters out the other sideband . It is a compromise
between AM and SSB , enabing simple receivers to be used , but requires almost as much transmitter power as AM . One advantage is only half the bandwidth of an
AM signal is used . It can be heard in the transmission of certain radio time signal stations . Vestigial sideband is used for over the air Television
Broadcasts both analog and digital .
</item 1>

<item 2>
Shortwave radio
See also ( edit ) ALLISS -- a very large rotatable antenna system used in international broadcasting Amateur radio -- also uses the shortwave bands , but with
power levels under 2 kW International broadcasting List of American shortwave broadcasters List of shortwave radio broadcasters Long wave Medium wave Shortwave
bands -- shortwave spectrum allocation shortwave relay station -- the fundamental way in which programmes are broadcast on shortwave SSB -- a method of radio
signal modulation References ( edit ) Jump up ^ Grundig Satellit 400 international / professional ShortWaveRadio.ch Jump up ^ Inconsistent article :
itwissen.info : KW ( Kurzwelle ) SW ( short wave ) Quote : `` ... Der KW - Frequenzbereich ( SW ) liegt zwischen 3 MHz und 30 MHz ( < - 1 . definition ) ...
Kurzwelle ist auch eine sendetechnische Bezeichnung für Rundfunk im Frequenzbereich zwischen 2 , 3 MHz und 26 , 1 MHz ( < - 2 .
</item 2>

<item 3>
Shortwave radio
The last WRC took place in 2007 . At WRC - 97 in 1997 , the following bands were allocated for international broadcasting . AM shortwave broadcasting channels
are allocated with a 5 kHz separation for traditional analog audio broadcasting . Metre Band Frequency Range Remarks 120 m 2.3 -- 2.495 MHz tropical band 90 m
3.2 -- 3.4 MHz tropical band 75 m 3.9 -- 4 MHz shared with the North American amateur radio 80m band 60 m 4.75 -- 5.06 MHz tropical band 49 m 5.9 -- 6.2 MHz 41
m 7.2 -- 7.6 MHz shared with the amateur radio 40m band 31 m 9.4 -- 9.9 MHz currently the most heavily used band 25 m 11.6 -- 12.2 MHz 22 m 13.57 -- 13.87 MHz
19 m 15.1 -- 15.8 MHz 16 m 17.48 -- 17.9 MHz 15 m 18.9 -- 19.02 MHz almost unused ,
</item 3>

<item 4>
Shortwave radio
Continuous wave ( CW ) is on - and - off keying of a carrier , used for Morse code communications and Hellschreiber facsimile - based teleprinter transmissions
. Narrow - band frequency modulation ( NBFM ) is used typically above 20 MHz . Because of the larger bandwidth required , NBFM is commonly used for VHF
communication . Regulations limit the bandwidth of a signal transmitted in the HF bands , and the advantages of frequency modulation are greatest if the FM
signal has a wide bandwidth . NBFM is limited to short - range transmissions due to the multiphasic distortions created by the ionosphere . Digital Radio
Mondiale ( DRM ) is a digital modulation for use on bands below 30 MHz . Radioteletype , fax , digital , slow - scan television and other systems use forms of
frequency - shift keying or audio subcarriers on a shortwave carrier . These generally require special equipment to decode , such as software on a computer
equipped with a sound card .
</item 4>

<item 5>
Shortwave radio
9 MHz 15 m 18.9 -- 19.02 MHz almost unused , could become a DRM band 13 m 21.45 -- 21.85 MHz 11 m 25.6 -- 26.1 MHz may be used for local DRM broadcasting
Although countries generally follow the table above , there may be small differences between countries or regions . For example , in the official bandplan of
the Netherlands , the 49 m band starts at 5.95 MHz , the 41 m band ends at 7.45 MHz , the 11 m band starts at 25.67 MHz , and the 120 , 90 and 60 m bands are
absent altogether . Additionally , international broadcasters sometimes operate outside the normal WRC - allocated bands or use off - channel frequencies . This
is done for practical reasons , or to attract attention in crowded bands ( 60m , 49m , 40m , 41m , 31m , 25m ) . The new digital audio broadcasting format for
shortwave DRM operates 10 kHz or 20 kHz channels .
</item 5>

<item 6>
Shortwave radio
Contents ( hide ) 1 Frequency classifications 2 History 2.1 Development 2.2 Amateur use of shortwave propagation 3 Propagation characteristics 4 Types of
modulation 5 Users 6 Shortwave broadcasting 6.1 Frequency allocations 6.2 Advantages 6.3 Disadvantages 7 Shortwave listening 8 Shortwave broadcasts and music 9
Shortwave 's future 10 See also 11 References 12 External links Frequency classifications ( edit ) The widest popular definition of the shortwave frequency
interval is the ITU Region 1 ( EU + Africa + Russia ... ) definition , and is the span 1.6 -- 30 MHz , just above the medium wave band , which ends
approximately at 1.6 MHz .
</item 6>

<item 7>
Shortwave radio
Shortwave radio received its name because the wavelengths in this band are relatively shorter than 200 m ( 1,500 kHz ) which marked the original upper limit of
the medium frequency band first used for radio communications . The broadcast medium wave band now extends above the 200 m / 1,500 kHz limit , and the amateur
radio 1.8 MHz -- 2.0 MHz band ( known as the `` top band '' ) is the lowest - frequency band considered to be ' shortwave ' . Early long distance radio
telegraphy used long waves , below 300 kilohertz ( kHz ) . The drawbacks to this system included a very limited spectrum available for long distance
communication , and the very expensive transmitters , receivers and gigantic antennas that were required . It was also difficult to beam the radio wave
directionally with long wave , resulting in a major loss of power over long distances . Prior to the 1920s , the shortwave frequencies above 1.5 MHz were
regarded as useless for long distance communication and were designated in many countries for amateur use .
</item 7>

<item 8>
Shortwave radio
Users ( edit ) Portable shortwave receiver 's digital display tuned to the 75 meter band Some established users of the shortwave radio bands may include :
International broadcasting primarily by government - sponsored propaganda , international news ( for example , the BBC World Service ) or cultural stations to
foreign audiences : the most common use of all . Domestic broadcasting : to widely dispersed populations with few longwave , mediumwave and FM stations serving
them ; or for specialty political , religious and alternative media networks ; or of individual commercial and non-commercial paid broadcasts . Oceanic air
traffic control uses the HF / shortwave band for long distance communication to aircraft over the oceans and poles , which are far beyond the range of
traditional VHF frequencies . Modern systems also include satellite communications , such as ADS - C / CPDLC `` Utility '' stations transmitting messages not
intended for the general public , such as merchant shipping , marine weather , and ship - to - shore stations ; for aviation weather and air - to - ground
communications ; for military communications ; for long - distance governmental purposes , and for other non-broadcast communications .
</item 8>

<item 9>
Shortwave radio
L band S band Digital systems ADR DAB - S DVB - SH S - DMB SDR Commercial radio providers 1worldspace Sirius XM Holdings SiriusXM Canada Codecs AAC AMR - WB+
HDC HE - AAC MPEG - 1 Audio Layer II Subcarrier signals AMSS DirectBand PAD RDS / RBDS SCA / SCMO DARC Related topics Technical ( audio ) Audio data compression
Audio signal processing Technical ( AM stereo formats ) Belar C - QUAM Harris Kahn - Hazeltine Magnavox Technical ( emission ) AM broadcasting AM expanded band
Cable radio Digital radio Error detection and correction FM broadcast band FM broadcasting Multipath propagation Shortwave relay station Cultural History of
radio International broadcasting Comparison of radio systems Radio spectrum ( ITU ) ELF 3 Hz / 100 Mm 30 Hz / 10 Mm SLF 30 Hz / 10 Mm 300 Hz / 1 Mm ULF 300 Hz /
1 Mm 3 kHz / 100 km VLF 3 kHz / 100 km 30 kHz / 10 km LF 30 kHz / 10 km 300 kHz / 1 km MF 300 kHz / 1 km 3 MHz / 100 m HF
</item 9>

<item 10>
Shortwave radio
`` Short - Wave Radio Telephone is Success in Tests '' Popular Mechanics , July 1931 , mid page experiments carried out for the French and British governments
Que Escuchar en la Onda Corta en Español website Time signal stations Longwave BPC BPL BSF DCF77 JJY RBU RJH66 ( Beta ) RTZ MSF TDF WWVB Shortwave BPM BSF CHU
HD2IOA HLA JN53DV RWM ROA WWV WWVH ( Hawaii , USA ) YVTO VHF / FM / UHF RDS CT Satellite GPS BeiDou Galileo GLONASS IRNSS QZSS Defunct HBG NAA WWVL OMA OLB5 VNG
Y3S Analog and digital audio broadcasting Terrestrial Radio modulation AM FM COFDM Frequency allocations LW ( LF ) MW ( MF ) SW ( HF ) VHF ( low / mid / high )
L band ( UHF ) Digital systems CAM - D DAB / DAB+ DRM / DRM+ FMeXtra HD Radio CDR DVB - T2 Lite Satellite Frequency allocations C band K band L band S band
Digital systems ADR DAB - S DVB - SH S - DMB
</item 10>
=== END ITEM id=db78ad82a0383821 ===

=== ITEM id=3b12a14890979b8e ===
Gold sentences and facts:
Sentence 0: Songwriter ( s ) Kirsty MacColl / Philip Rambow
  Fact 0.0: The songwriters of the song are Kirsty MacColl and Philip Rambow.
Sentence 1: `` There 's a Guy Works Down the Chip Shop , Swears He 's Elvis '' Single by Kirsty MacColl from the album Desperate Character A-side `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' B - side `` Hard To Believe There 's a Guy ... ( country version ) '' Released May 1981 Format 7 '' Vinyl Genre Country pop Label Polydor Songwriter ( s ) Kirsty MacColl / Philip Rambow Producer ( s ) Barry Farmer Kirsty MacColl singles chronology `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop , Swears He 's Elvis '' ( 1981 ) `` See That Girl '' ( 1981 ) `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' ( 1981 ) `` See That Girl '' ( 1981 )
  Fact 1.0: 'There's a Guy Works Down the Chip Shop, Swears He's Elvis' is a single by Kirsty MacColl.
  Fact 1.1: 'There's a Guy Works Down the Chip Shop, Swears He's Elvis' is from the album Desperate Character.
  Fact 1.2: The single's A-side is 'There's a Guy Works Down the Chip Shop Swears He's Elvis'.
  Fact 1.3: The single's B-side is 'Hard To Believe There's a Guy ... (country version)'.
  Fact 1.4: The single was released in May 1981.
  Fact 1.5: The single's format is 7-inch vinyl.
  Fact 1.6: The single's genre is country pop.
  Fact 1.7: The single's label is Polydor.
  Fact 1.8: The single's songwriters are Kirsty MacColl and Philip Rambow.
  Fact 1.9: The single's producer is Barry Farmer.
  Fact 1.10: The Kirsty MacColl singles chronology includes 'Keep Your Hands Off My Baby' (1981).
  Fact 1.11: The Kirsty MacColl singles chronology includes 'There's a Guy Works Down the Chip Shop, Swears He's Elvis' (1981).
  Fact 1.12: The Kirsty MacColl singles chronology includes 'See That Girl' (1981).
  Fact 1.13: The Kirsty MacColl singles chronology includes 'Keep Your Hands Off My Baby' (1981) a second time.
  Fact 1.14: The Kirsty MacColl singles chronology includes 'There's a Guy Works Down the Chip Shop Swears He's Elvis' (1981).
  Fact 1.15: The Kirsty MacColl singles chronology includes 'See That Girl' (1981) a second time.

Retrieved items:
<item 1>
Neil Diamond
He did songwriting wherever he could , including on buses , and used an upright piano above the Birdland Club in New York City . One of the causes of this early
nomadic life as a songwriter was due to his songs having too many words : `` I 'd spent a lot of time on lyrics , and they were looking for hooks , and I did
n't really understand the nature of that , '' he says . During those lean years , he was only able to sell about one song a week , barely enough to survive on .
He found himself only earning enough to spend 35 cents a day on food ( US $3 in 2017 dollars ) . However , the privacy he had above the Birdland Club allowed
him to focus on writing without distractions ; as he explained , `` Something new began to happen . I was n't under the gun , and suddenly interesting songs
began to happen , songs that had things none of the others did . '' Among them were `` Cherry , Cherry '' and `` Solitary Man '' . `` Solitary Man '' was the
first record that Diamond recorded in his own name that made the charts .
</item 1>

<item 2>
There's a Guy Works Down the Chip Shop Swears He's Elvis
There 's a Guy Works Down the Chip Shop Swears He 's Elvis - wikipedia There 's a Guy Works Down the Chip Shop Swears He 's Elvis Jump to : navigation , search
This article relies too much on references to primary sources . Please improve this by adding secondary or tertiary sources . ( March 2008 ) ( Learn how and
when to remove this template message ) `` There 's a Guy Works Down the Chip Shop , Swears He 's Elvis '' Single by Kirsty MacColl from the album Desperate
Character A-side `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' B - side `` Hard To Believe There 's a Guy ... ( country version ) '' Released
May 1981 Format 7 '' Vinyl Genre Country pop Label Polydor Songwriter ( s ) Kirsty MacColl / Philip Rambow Producer ( s ) Barry Farmer Kirsty MacColl singles
chronology `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop ,
</item 2>

<item 3>
There's a Guy Works Down the Chip Shop Swears He's Elvis
Track listing ( edit ) `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' ( K. MacColl / P. Rambow ) `` Hard to Believe '' ( K. MacColl ) `` There
's a Guy Works Down the Chip Shop Swears He 's Elvis ( country version ) '' ( K. MacColl / P. Rambow ) References ( edit ) Jump up ^ `` There 's a Guy Works
Down the Chip Shop Swears He 's Elvis - Kirsty MacColl '' . Retrieved 2014 - 10 - 30 . Jump up ^ `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis 7
'' single - Kirsty MacColl `` . Retrieved 2014 - 10 - 30 .
</item 3>

<item 4>
All I Want for Christmas Is My Two Front Teeth
The song was introduced in 1948 by singing group The Satisfiers on Perry Como 's radio show , and originally recorded by Spike Jones & His City Slickers on
December 6 , 1947 , with lead vocal by George Rock . That version reached the top of the pop charts in 1948 , and again in 1949 . The song has also been
recorded by numerous other singers and performers , including Ray Stevens , George Strait , Danny Kaye with The Andrews Sisters , Urbie Green , The Platters ,
Dread Zeppelin , The Kelly Family , Nat King Cole ( reportedly Gardner 's favorite version ) , The Chipmunks , the Hampton String Quartet , The Three Stooges
and Count von Count of Sesame Street . In the versions by Spike , The Chipmunks ( Theodore on lead vocals ) , and Ray Stevens , the song ends with the performer
shouting the words `` Happy New Year ! ! ! '' Homer and Jethro parodied the song as `` All I Want for Christmas Is My Upper Plate '' , on their 1968 album Cool
Crazy Christmas .
</item 4>

<item 5>
The Rolling Stones
Later in 1963 Oldham and Easton arranged the band 's first big UK concert tour as a supporting act for American stars including Bo Diddley , Little Richard and
the Everly Brothers . The tour gave the band the opportunity to hone their stagecraft . During the tour the band recorded their second single , a Lennon --
McCartney - penned number entitled `` I Wanna Be Your Man '' ; it reached No. 12 in the UK charts . `` I Wanna Be Your Man '' was written and given to the
Stones when John Lennon and Paul McCartney visited them in the studio as the two Beatles liked to give the copyrights to songs away to their friends . A Beatles
version of the song was also recorded and released on the 1963 album With the Beatles . The third single by the Stones , Buddy Holly 's `` Not Fade Away '' ,
itself based on Bo Diddley 's style , was released in February 1964 and reached No. 3 . Oldham saw little future for an act that lost significant songwriting
royalties by playing songs of what he described as `` middle - aged blacks '' , limiting the appeal to teenage audiences .
</item 5>

<item 6>
The Rolling Stones
Jump up ^ Haslam 2015 , p. 91 . Jump up ^ Wyman 2002 , pp. 65 . Jump up ^ Wyman 2002 , pp. 80 -- 83 . Jump up ^ Fricke , David ( 17 April 2008 ) . `` Blues
Brothers '' . Rolling Stone . Archived from the original on 5 April 2008 . Jump up ^ Gilliland 1969 , show 30 , track 2 . Jump up ^ `` Yes , The Beatles Once
Wrote a Song for the Rolling Stones '' . Ultimate Classic Rock . Archived from the original on 9 July 2017 . Retrieved 15 August 2017 . Jump up ^ `` How The
Beatles ended up writing `` I Wanna Be Your Man '' for The Rolling Stones `` . The Beatles . Archived from the original on 15 August 2017 . Retrieved 15 August
2017 . ^ Jump up to : `` UK Charts -- The Rolling Stones '' . Official Charts Company . Archived from the original on 20 October 2014 .
</item 6>

<item 7>
There's a Guy Works Down the Chip Shop Swears He's Elvis
Retrieved 2014 - 10 - 30 . Jump up ^ Information at Svensk mediedatabas External links ( edit ) freeworld - Kirsty MacColl fansite Kirsty MacColl Studio albums
Desperate Character Kirsty MacColl Kite Electric Landlady Titanic Days Tropical Brainstorm Compilations Galore Notable singles `` They Do n't Know '' `` Keep
Your Hands Off My Baby '' `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' `` You Still Believe in Me '' `` Terry '' `` A New England '' `` Days
'' `` Do n't Come The Cowboy With Me Sonny Jim ! ''
</item 7>

<item 8>
There's a Guy Works Down the Chip Shop Swears He's Elvis
`` Walking Down Madison '' `` Soho Square '' With The Pogues `` Fairytale of New York '' Related articles Ewan MacColl Steve Lillywhite Retrieved from ``
https://en.wikipedia.org/w/index.php?title=There%27s_a_Guy_Works_Down_the_Chip_Shop_Swears_He%27s_Elvis&oldid=799608311 '' Categories : 1981 singles 1981 songs
Elisabeth Andreassen songs Kirsty MacColl songs Songs written by Kirsty MacColl Songs about Elvis Presley Polydor Records singles Hidden categories : Articles
lacking reliable references from March 2008 All articles lacking reliable references Talk About Wikipedia Svenska Edit links This page was last edited on 8
September 2017 , at 19 : 12 . About Wikipedia
</item 8>

<item 9>
There's a Guy Works Down the Chip Shop Swears He's Elvis
Swears He 's Elvis '' ( 1981 ) `` See That Girl '' ( 1981 ) `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop Swears He 's
Elvis '' ( 1981 ) `` See That Girl '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' was Kirsty MacColl 's first hit single ,
charting in the UK at # 14 in 1981 , and remaining in the charts for nine weeks . It reached # 9 in the Irish charts . A promotional version was released in the
USA , which instead referenced a truck stop as opposed to the very British chip shop , but this was only available to DJs and did not get a full release .
Norwegian singer Elisabeth Andreasson covered the song on her 1981 country album Angel of the Morning , with lyrics in Swedish by Hasse Olsson as `` Killen ner
' på Konsum svär att han är Elvis '' ( `` The guy down Konsum '' swears he 's Elvis ) .
</item 9>

<item 10>
Sacrifice (song)
Jump up ^ `` UK certifications , database '' . Bpi . Retrieved 9 September 2008 . External links ( edit ) Lyrics of this song at MetroLyrics Elton John songs
1960s and 1970s `` Skyline Pigeon '' ( 1968 ) `` I 've Been Loving You '' ( 1968 ) `` Lady Samantha '' ( 1969 ) `` It 's Me That You Need '' ( 1969 ) `` Border
Song '' ( 1970 ) `` Rock and Roll Madonna '' ( 1970 ) `` From Denver to L.A. '' ( 1970 ) `` Take Me to the Pilot '' ( 1970 ) `` Your Song '' ( 1970 ) `` Friends
'' ( 1971 ) `` Levon '' ( 1971 ) `` Tiny Dancer '' ( 1972 ) `` Rocket Man '' ( 1972 ) `` Honky Cat '' ( 1972 ) `` I Think I 'm Going to Kill Myself '' ( 1972 )
`` Mona Lisas and Mad Hatters '' ( 1972 ) `` Crocodile Rock '' ( 1972 ) `` Elderberry Wine '' (
</item 10>
=== END ITEM id=3b12a14890979b8e ===

=== ITEM id=12102aeee8330dff ===
Gold sentences and facts:
Sentence 0: The Raiders had a great chance to score a touchdown early in the game after cornerback Charles Woodson intercepted Buccaneers quarterback Brad Johnson 's pass on the third play of the game and returned it 12 yards to the Tampa Bay 36 - yard line . However , six plays later , Tampa Bay defensive end Simeon Rice sacked Raiders quarterback Rich Gannon on third down , forcing Oakland to settle for kicker Sebastian Janikowski 's 40 - yard field goal to give them a 3 -- 0 lead .
  Fact 0.0: Cornerback Charles Woodson intercepted a pass thrown by Buccaneers quarterback Brad Johnson.
  Fact 0.1: The interception happened on the third play of the game.
  Fact 0.2: Charles Woodson returned the interception 12 yards to the Tampa Bay 36-yard line.
  Fact 0.3: The Raiders had a great chance to score a touchdown early in the game after the interception.
  Fact 0.4: Six plays after the interception, Tampa Bay defensive end Simeon Rice sacked Raiders quarterback Rich Gannon on third down.
  Fact 0.5: The sack forced Oakland to settle for a field goal.
  Fact 0.6: Kicker Sebastian Janikowski kicked a 40-yard field goal for Oakland.
  Fact 0.7: The field goal gave Oakland a 3-0 lead.
Sentence 1: Still , even Gruden had trouble getting the offense in sync during his first year as Tampa Bay 's head coach . In 2002 , the Buccaneers ranked 25th in the league in total yards gained ( 5,222 ) . Quarterback Brad Johnson made the Pro Bowl , completing 281 out of 451 passes for 3,049 yards , 22 touchdowns , and only 6 interceptions . Running back Michael Pittman led the team in rushing with 718 yards and one touchdown , and caught 59 passes for 477 yards . Pro Bowl fullback Mike Alstott had 548 rushing yards and 5 touchdowns , and also had 35 receptions for 242 yards and 2 touchdowns . Wide receiver Keyshawn Johnson led the team with 76 receptions for 1,088 yards and 5 touchdowns , while wide receiver Keenan McCardell had 61 receptions for 670 yards and 6 touchdowns .
  Fact 1.0: Even Gruden had trouble getting the offense in sync during his first year as Tampa Bay's head coach.
  Fact 1.1: In 2002, the Buccaneers ranked 25th in the league in total yards gained.
  Fact 1.2: The Buccaneers gained 5,222 total yards in 2002.
  Fact 1.3: Quarterback Brad Johnson made the Pro Bowl.
  Fact 1.4: Brad Johnson completed 281 of 451 passes for 3,049 yards.
  Fact 1.5: Brad Johnson threw 22 touchdowns.
  Fact 1.6: Brad Johnson threw only 6 interceptions.
  Fact 1.7: Running back Michael Pittman led the Buccaneers in rushing with 718 yards and one touchdown.
  Fact 1.8: Michael Pittman caught 59 passes for 477 yards.
  Fact 1.9: Pro Bowl fullback Mike Alstott had 548 rushing yards and 5 touchdowns.
  Fact 1.10: Mike Alstott had 35 receptions for 242 yards and 2 touchdowns.
  Fact 1.11: Wide receiver Keyshawn Johnson led the team with 76 receptions for 1,088 yards and 5 touchdowns.
  Fact 1.12: Wide receiver Keenan McCardell had 61 receptions for 670 yards and 6 touchdowns.

Retrieved items:
<item 1>
Super Bowl XXXVII
Gurley 91 Chuck Darby 92 Anthony McFarland 93 DeVone Claybrooks 94 Greg Spires 95 Ron Warner 96 Ellis Wyms 97 Simeon Rice 98 Corey Smith 99 Warren Sapp -- Onome
Ojo -- T.T. Toliver Head coach : Jon Gruden Coaches : Joe Barry Jeremy Bates Richard Bisaccia Michael Christianson Jay Gruden Monte Kiffin Richard Mann Rod
Marinelli Raheem Morris Bill Muir Johnny Parker Stan Parrish Mike Tomlin Art Valero Kirby Wilson Tampa Bay Buccaneers Founded in 1976 Based and headquartered in
Tampa , Florida Franchise Franchise Seasons History Coaches Players Expansion draft First - round draft picks Draft history Starting quarterbacks Broadcasters
Stadiums Tampa Stadium Raymond James Stadium Culture and lore Hugh Culverhouse Malcolm Glazer Tampa 2 Captain Fear Cheerleaders Snow Bowl A Triumph of the Heart
: The Ricky Bell Story Rivalries Carolina Panthers New Orleans Saints Miami Dolphins Playoff appearances ( 10 ) 1979 1981 1982 1997 1999 2000 2001 2002 2005
2007 Division championships ( 6 ) 1979 1981 1999
</item 1>

<item 2>
Super Bowl XXXVII
Aided with Pittman 's gains of 6 and 19 yards , the Buccaneers scored their first touchdown on a 2 - yard run from fullback Mike Alstott , increasing their lead
to 13 -- 3 . Then with 3 : 45 left in the half , Tampa Bay drove 77 yards , assisted by a pair of catches by Alstott for 28 total yards . Johnson finished the
drive with a 5 - yard touchdown pass to wide receiver Keenan McCardell to give the Buccaneers a 20 -- 3 halftime lead . Third quarter ( edit ) Tampa Bay
continued to dominate the game for most of the third quarter . The Buccaneers forced the Raiders to punt on the opening drive of the second half . Next , Tampa
Bay marched 89 yards on a 14 - play drive that took 7 : 52 off the clock , and ended with Johnson 's 8 - yard touchdown pass to McCardell to increase their lead
to 27 -- 3 .
</item 2>

<item 3>
Super Bowl XXXVII
Defensive back Tory James was also a big contributor with 4 interceptions . Playoffs ( edit ) Further information : 2002 -- 03 NFL playoffs The second - seeded
Buccaneers defeated the fourth - seeded San Francisco 49ers , 31 -- 6 , and the top - seeded Philadelphia Eagles on the road , 27 -- 10 , in the playoffs , to
make the Super Bowl for the first time in franchise history . Meanwhile , the top - seeded Raiders were victorious against the fourth - seeded New York Jets ,
30 -- 10 , and the second - seeded Tennessee Titans , 41 -- 24 . The Raiders won against the Titans through Gannon 's pass - oriented offense . Super Bowl
pre-game News ( edit ) The Raiders entered the game favored to win in their first Super Bowl in 19 years . They were also the first franchise to appear in the
Super Bowl in four different decades ( 1960s , 1970s , 1980s , 2000s ; the 1990s was the only decade that they did n't appear in the Super Bowl ) . However ,
much of the media hype surrounded the Gruden trade prior to the season .
</item 3>

<item 4>
Super Bowl XXXVII
Then on third and 18 from the Oakland 29 - yard line , Tampa Bay linebacker Derrick Brooks intercepted a pass from Gannon and returned it 44 yards for a
touchdown , giving the Buccaneers a 41 -- 21 lead with only 1 : 18 left , and leading Buccaneers radio announcer Gene Deckerhoff to make his famous call of ``
The dagger 's in , we 're gon na win the Super Bowl ! '' . A few plays later , with the Raiders now playing for pride , Dwight Smith intercepted a tipped pass
and returned it 50 yards for a touchdown with only two seconds remaining . Gramatica kicked the extra point , and the Buccaneers were up 48 -- 21 . Raiders
defensive tackle Chris Cooper returned Gramatica 's kickoff 6 yards before being tackled by Jack Golden , ending the game . The Buccaneers won their first -
ever Super Bowl with a 48 - 21 victory . Gannon said after the game that his performance was `` nightmarish '' .
</item 4>

<item 5>
Super Bowl XXXVII
But that changed when the team hired Tony Dungy as head coach in 1996 . Dungy , along with his defensive coordinator Monte Kiffin , rebuilt the defense around a
core group of players , such as defensive lineman Warren Sapp , linebacker Derrick Brooks , and defensive backs Ronde Barber and John Lynch . By 1997 , Tampa
Bay ranked 3rd in the league in total yards allowed , 2nd in 1998 , and 3rd in 2000 . With one of the best defenses in the league , Dungy helped guide the
Buccaneers to four playoff appearances in his six years as the team 's head coach . But the team always had one of the worst offenses in the league , and this
was a major factor in their playoff losses . Thus , Dungy was fired and replaced by Gruden . Still , even Gruden had trouble getting the offense in sync during
his first year as Tampa Bay 's head coach . In 2002 , the Buccaneers ranked 25th in the league in total yards gained ( 5,222 ) .
</item 5>

<item 6>
Super Bowl XXXVII
The Raiders came into the game as four - point favorites . However , the Tampa Bay defense dominated the contest . Oakland quarterback Rich Gannon threw a Super
Bowl record five interceptions , three of which were returned for touchdowns . The Buccaneers also sacked Gannon five times , and scored 34 consecutive points
to build a 34 -- 3 lead late in the third quarter . Tampa Bay safety Dexter Jackson , who had two of those interceptions and returned them for 34 yards , was
named Super Bowl MVP. Jackson became only the second safety and third defensive back named Super Bowl MVP .
</item 6>

<item 7>
Super Bowl XXXVII
This was the last Super Bowl until Super Bowl 50 to take place in California . It was the first and only time until Super Bowl XLVIII in Super Bowl history that
the league 's number one - ranked offense ( Raiders ) was pitted against the league 's number one - ranked defense ( Buccaneers ) . Super Bowl XXXVII is also
referred to as the `` Pirate Bowl '' , due to both teams ' pirate - themed names . The game is sometimes referred to as the `` Gruden Bowl '' , because the
primary storyline surrounding the game revolved around Jon Gruden . Gruden was the Raiders ' head coach from 1998 to 2001 , and then became the Buccaneers coach
in 2002 . Tampa Bay , `` Gruden 's new team '' , made their first Super Bowl appearance in team history after posting a 12 -- 4 regular season record . Oakland
, `` Gruden 's old team '' , advanced to their fifth Super Bowl after an 11 -- 5 regular season . This was also the first Super Bowl to feature two former
division rivals , as the Buccaneers and Raiders both played in the AFC West in 1976 .
</item 7>

<item 8>
Super Bowl XXXVII
Quarterback Brad Johnson made the Pro Bowl , completing 281 out of 451 passes for 3,049 yards , 22 touchdowns , and only 6 interceptions . Running back Michael
Pittman led the team in rushing with 718 yards and one touchdown , and caught 59 passes for 477 yards . Pro Bowl fullback Mike Alstott had 548 rushing yards and
5 touchdowns , and also had 35 receptions for 242 yards and 2 touchdowns . Wide receiver Keyshawn Johnson led the team with 76 receptions for 1,088 yards and 5
touchdowns , while wide receiver Keenan McCardell had 61 receptions for 670 yards and 6 touchdowns . However , the Buccaneers ' defense was still the strength
of the team , leading the NFL in total defense ( 252.8 yards per game ) , pass defense ( 155.6 yards per game ) , points allowed ( 12.3 points per game ) ,
passing touchdowns allowed ( 10 ) , interceptions ( 31 ) , and opponent passer rating ( 48.4 ) . Brooks , Lynch , Sapp , and defensive end Simeon Rice all had
Pro Bowl years .
</item 8>

<item 9>
Super Bowl XXXVII
January 15 , 2003 . Retrieved August 27 , 2012 . Jump up ^ Mitchell , Fred ( January 27 , 2003 ) . `` In their fashion , Bucs fans laugh '' . Chicago Tribune .
Retrieved August 27 , 2012 . Jump up ^ Vivinetto , Gina ( January 27 , 2003 ) . `` Super Bowl XXXVII : Superstars belt out favorites , or lip synch them '' .
St. Petersburg Times . ^ Jump up to : George , Thomas ( January 27 , 2003 ) . `` Bucs steal the treasure '' . The New York Times . Retrieved August 27 , 2012 .
^ Jump up to : `` Tampa Bay Buccaneers 48 vs. Oakland Raiders 21 '' . pro-football - reference . Retrieved August 27 , 2012 . Jump up ^ Cunningham , Michael ;
Skolnick , Ethan J. ( January 27 , 2003 ) . `` Treu Gets The Call In Robbins ' Absence '' . South Florida Sun - Sentinel .
</item 9>

<item 10>
Super Bowl XXXVII
Better still for the Buccaneers was that Oakland had n't changed their audible - calling signals that Gruden himself had installed , thus tipping off plays
repeatedly . Johnson finished the game with 18 out of 34 completions for 215 yards and 2 touchdowns , with 1 interception , along with 10 rushing yards .
Pittman was the top rusher of the game with 129 yards . Alstott was the game 's second leading rusher with 15 yards and a touchdown , and had 5 receptions for
43 yards . Wide receiver Keyshawn Johnson was the Buccaneers ' leading receiver with 6 catches for 69 yards . Smith recorded 2 interceptions , 94 return yards ,
and 2 touchdowns . He also added another 23 yards on a kickoff return . Gannon finished the game 24 out of 44 for 272 yards and 2 touchdowns , but was
intercepted a Super Bowl record 5 times . Garner was their leading rusher , but with only 10 yards , and caught 7 passes for 51 yards . Rice was the Raiders '
leading receiver of the game with 5 catches for 77 yards and a touchdown .
</item 10>
=== END ITEM id=12102aeee8330dff ===

=== ITEM id=d0da9e41c51a1698 ===
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
Top Adult Contemporary : 1961 - 2001 . Record Research . p. 200 . Jump up ^ `` Top Singles -- Volume 26 , No. 14 & 15 , January 08 1977 '' . RPM . Library and
Archives Canada . Retrieved June 13 , 2016 . Jump up ^ `` Top 100 Hits of 1976 / Top 100 Songs of 1976 '' . Musicoutfitters.com . Retrieved 2016 - 10 - 12 .
Jump up ^ Erlewine , Michael . The Man with the Sad Face -- Review at AllMusic . Retrieved April 16 , 2015 . External links ( edit ) Lyrics of this song at
MetroLyrics ( hide ) Lou Rawls Albums Stormy Monday All Things in Time Unmistakably Lou When You Hear Lou ,
</item 1>

<item 2>
You'll Never Find Another Love Like Mine
You 'll Never Find Another Love Like Mine - wikipedia You 'll Never Find Another Love Like Mine Jump to : navigation , search `` You 'll Never Find Another Love
Like Mine '' Single by Lou Rawls from the album All Things in Time B - side `` Let 's Fall in Love All Over Again '' Released May 1976 Format 7 '' Genre R&B ,
jazz , disco Length 4 : 26 ( album version ) 3 : 36 ( single edit ) Label Philadelphia International Records Songwriter ( s ) Kenny Gamble Leon Huff Producer (
s ) Kenny Gamble Leon Huff Lou Rawls singles chronology `` A Natural Man '' ( 1971 ) `` You 'll Never Find Another Love Like Mine '' ( 1976 ) `` Groovy People
'' ( 1976 ) `` A Natural Man '' ( 1971 ) `` You 'll Never Find Another Love Like Mine '' ( 1976 ) `` Groovy People '' ( 1976 ) `` You 'll Never Find Another
Love Like Mine '' ( written by Kenny Gamble & Leon Huff ) is a song performed by R&B singer Lou Rawls on his 1976 album All Things in Time .
</item 2>

<item 3>
You'll Never Find Another Love Like Mine
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 3>

<item 4>
You'll Never Find Another Love Like Mine
Contents ( hide ) 1 Chart performance 1.1 Weekly charts 1.2 Year - end charts 2 Later uses 3 Cover versions 4 See also 5 References 6 External links Chart
performance ( edit ) Weekly charts ( edit ) Chart ( 1976 -- 77 ) Peak position Australia KMR 22 Canada RPM Top Singles UK 10 US Billboard Hot 100 US Billboard
Hot Dance Club Play US Billboard Hot Adult Contemporary Tracks US Billboard Hot Black Singles Year - end charts ( edit ) Chart ( 1976 ) Rank Canada 38 US
Billboard Hot 100 32 Chart ( 1977 ) Rank Australia 97 Later uses ( edit ) It can also be heard in the films The Hot Chick ( 2002 ) , Guess Who ( 2005 ) ,
Disturbia ( 2007 ) , Ice Age : Dawn of the Dinosaurs ( 2009 ) , Hit and Run ( 2012 ) , Veronica Mars ( 2014 ) , and on the TV sitcoms My Wife and Kids ( 2001 --
2005 ) , in which Lou Rawls himself sings it to Damon Wayans during a colonoscopy ,
</item 4>

<item 5>
You'll Never Find Another Love Like Mine
Record Research . p. 213 . Jump up ^ `` Billboard Hot 100 '' . 1976 - 09 - 04 . Retrieved 2015 - 06 - 07 . Jump up ^ `` Billboard Hot 100 '' . 1976 - 09 - 11 .
Retrieved 2015 - 06 - 07 . Jump up ^ `` The Hits Just Keep On Comin ' : You 'll Never Find Another Love Like Mine '' . Hitsjustkeeponcomin.blogspot.com. 2005 -
01 - 12 . Retrieved 2016 - 10 - 12 . Jump up ^ ( 1 ) Jump up ^ `` Archived copy '' . Archived from the original on June 2 , 2016 . Retrieved October 27 , 2015 .
Jump up ^ `` Image : RPM Weekly - Library and Archives Canada '' . Bac-lac.gc.ca . Retrieved 2016 - 10 - 12 . Jump up ^ Whitburn , Joel ( 2002 ) . Top Adult
Contemporary : 1961 - 2001 . Record Research .
</item 5>

<item 6>
You'll Never Find Another Love Like Mine
The song proved to be Rawls ' breakthrough hit , reaching number one on both the R&B and Easy Listening charts as well as number four on the dance chart and
number two on the US Billboard Hot 100 , where it was kept from the top spot for two weeks by `` You Should Be Dancing '' by The Bee Gees and `` ( Shake , Shake
, Shake ) Shake Your Booty '' by KC and the Sunshine Band , respectively . This was the first and only time that one of Rawls ' records managed to reach
Billboard 's pop top ten . It was the first big hit for Philadelphia International to feature the reformulated MFSB , after many of the original members left
Gamble and Huff for better opportunities . The song started Rawls ' live shows from 1977 on . The single went on to sell over a million copies and was certified
gold by the RIAA . Contents ( hide ) 1 Chart performance 1.1 Weekly charts 1.
</item 6>

<item 7>
For What It's Worth
If I Could Turn Back Time '' `` Just Like Jesse James '' 1990s `` Heart of Stone '' `` You Would n't Know Love '' `` Baby I 'm Yours '' `` The Shoop Shoop Song
( It 's in His Kiss ) '' `` Love and Understanding '' `` Save Up All Your Tears '' `` Love Hurts '' `` Could 've Been You '' `` When Lovers Become Strangers ''
`` Oh No Not My Baby '' `` Whenever You 're Near '' `` Many Rivers to Cross '' `` Walking in Memphis '' `` One by One '' `` Not Enough Love in the World '' ``
The Sun Ai n't Gonna Shine Anymore '' `` Paradise Is Here '' `` Believe '' `` Strong Enough '' `` All or Nothing '' `` Dov'è l'amore '' 2000s `` The Music 's No
Good Without You '' `` Song for the Lonely '' `` Alive Again '' `` A Different Kind of Love Song '' `` When the Money 's Gone '' `` Love One Another '' 2010s ``
You Have n't Seen the Last of Me '' `` Woman 's World '' `` I Hope You Find It '' `` Take It Like a Man '' `` I Walk Alone '' Guest singles
</item 7>

<item 8>
You'll Never Find Another Love Like Mine
in which Lou Rawls himself sings it to Damon Wayans during a colonoscopy , That ' 70s Show ( 1998 -- 2006 ) , South Park ( 1997 -- ) , Two and a Half Men ( 2003
-- 2015 ) , and Psych ( 2006 -- 2014 ) . The song could also be heard in an episode of The Proud Family , where Lou Rawls sings and dances with Penny Proud . In
addition , it was performed by the singer in doll form in an Action League Now ! segment on KaBlam ! . The song `` If You Could Love Me '' by Edwyn Collins uses
many of the same chords , and has a similar sound . Rocco DiSpirito and Karina Smirnoff also danced the rumba to this song in week 2 of season 7 of Dancing with
the Stars .
</item 8>

<item 9>
You'll Never Find Another Love Like Mine
Cover versions ( edit ) `` You 'll Never Find Another Love Like Mine '' has been covered by singers Michael Bublé , Laura Pausini , reggae singer John Holt and
by the Dub Pistols ( who use a sample of John Holt 's version ) on their Speakers and Tweeters album . Jazz saxophonist Stanley Turrentine covered the song on
his 1976 album The Man with the Sad Face . as did Hank Crawford on Hank Crawford 's Back . This song was also featured on The Proud Family Soundtrack . See also
( edit ) List of number - one R&B singles of 1976 ( U.S. ) List of number - one adult contemporary singles of 1976 ( U.S. ) References ( edit ) Jump up ^
Whitburn , Joel ( 2004 ) . Top R&B / Hip - Hop Singles : 1942 - 2004 . Record Research . p. 483 . Jump up ^ Whitburn , Joel ( 2004 ) . Hot Dance / Disco : 1974
- 2003 . Record Research . p. 213 .
</item 9>

<item 10>
You'll Never Find Another Love Like Mine
You 've Heard It All Let Me Be Good to You Sit Down and Talk to Me Singles `` Love Is a Hurtin ' Thing '' `` Your Good Thing ( Is About to End ) '' `` Bring It
On Home to Me '' `` You 'll Never Find Another Love Like Mine '' `` Lady Love '' Retrieved from ``
https://en.wikipedia.org/w/index.php?title=You%27ll_Never_Find_Another_Love_Like_Mine&oldid=803225603 '' Categories : 1976 singles 1976 songs Billboard Adult
Contemporary number - one singles Billboard Hot R&B / Hip - Hop Songs number - one singles Lou Rawls songs Philadelphia International Records singles Songs
written by Kenny Gamble Songs written by Leon Huff Songs from animated films Hidden categories : All articles with dead external links Articles with dead
external links from July 2016 Articles with permanently dead external links Articles with hAudio microformats Talk Contents About Wikipedia Italiano Português
Edit links This page was last edited on 1 October 2017 , at 07 : 04 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional
terms may apply .
</item 10>
=== END ITEM id=d0da9e41c51a1698 ===
